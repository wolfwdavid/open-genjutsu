#!/usr/bin/env python
"""
export_workflow.py -- write the Wan2.2-Animate replacement graph that wan_animate.py sends to
ComfyUI as a workflow file you can open in the ComfyUI editor.

Two files are written next to this script:
  wan_animate_replace.api.json   API-format prompt (exactly what wan_animate.py POSTs to /prompt,
                                 for one 33-frame window with no continue_motion)
  wan_animate_replace.json       editor-format workflow (nodes, links, positions) built from the
                                 same graph; drag it onto the ComfyUI canvas
  wan_animate_replace_cloud.api.json
                                 the same window as run_cloud.py submits to a Comfy developer
                                 platform deployment: inputs are video clips and an image
                                 (uploaded by `comfy deploy run`), output is one mp4

The editor format needs the node definitions (widget order) from a running ComfyUI:
  python workflows/export_workflow.py --comfy http://127.0.0.1:8188
Without a server only the API-format file is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import wan_animate  # noqa: E402

PLACEHOLDER_DIR = "<WORK>/wan/<name>/in"


_ARGS = dict(
        unet="Wan2.2-Animate-14B-Q4_K_M.gguf",
        text_encoder="umt5-xxl-encoder-Q5_K_M.gguf",
        clip_vision="clip_vision_h.safetensors",
        vae="wan_2.1_vae.safetensors",
        shift=8.0,
        prompt="A real orange and white tabby cat wearing black sunglasses, sitting upright on a "
               "wooden stool, rapping into a hanging studio microphone, fixed camera.",
        negative=wan_animate.NEG_PROMPT,
        ref_name="reference.png",
        width=768, height=576,
        continue_frames=5,
        seed=20260925, steps=6, cfg=1.0, sampler="euler", scheduler="simple",
        decode_tile=512, decode_frames=64,
        name="cat",
)
_MODELS = {"loras": [("lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 1.0),
                     ("WanAnimate_relight_lora_fp16.safetensors", 1.0)]}


def build_api_graph() -> dict:
    a = types.SimpleNamespace(**_ARGS)
    win = dict(index=0, length=33, seq_start=0,
               pose_dir=f"{PLACEHOLDER_DIR}/pose", bg_dir=f"{PLACEHOLDER_DIR}/bg",
               mask_dir=f"{PLACEHOLDER_DIR}/mask", face_dir=f"{PLACEHOLDER_DIR}/face")
    nodes, _ = wan_animate.build_window(a, win, _MODELS, None)
    return nodes


def build_cloud_graph() -> dict:
    import run_cloud
    a = types.SimpleNamespace(**_ARGS, clip_fps=12.0, clip_crf=4)
    win = dict(index=0, length=33, seq_start=0, prev_count=0)
    clips = {k: Path(f"<asset-root>/{k}.mp4") for k in ("pose", "bg", "mask", "face")}
    clips["ref"] = Path("<asset-root>/ref.png")
    return run_cloud.build_window_cloud(a, win, _MODELS, clips, None)


def fetch_object_info(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/object_info", timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def to_editor_format(api: dict, info: dict) -> dict:
    """Convert an API prompt to the editor's LiteGraph JSON (nodes + links)."""
    # topological order for a left-to-right layout
    order: list[str] = []
    seen: set[str] = set()

    def visit(nid: str):
        if nid in seen:
            return
        seen.add(nid)
        for v in api[nid]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in api:
                visit(v[0])
        order.append(nid)

    for nid in api:
        visit(nid)

    depth: dict[str, int] = {}
    for nid in order:
        d = 0
        for v in api[nid]["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0] in api:
                d = max(d, depth[v[0]] + 1)
        depth[nid] = d
    col_y: dict[int, int] = {}

    nodes, links = [], []
    link_id = 0
    out_slots: dict[tuple[str, int], list] = {}
    for nid in order:
        n = api[nid]
        ct = n["class_type"]
        spec = info.get(ct)
        if spec is None:
            raise SystemExit(f"node type {ct} is not registered on this ComfyUI (missing custom node?)")
        req = spec["input"].get("required", {})
        opt = spec["input"].get("optional", {})
        ordered = list(req.items()) + list(opt.items())
        inputs_json, widgets = [], []
        for name, decl in ordered:
            kind = decl[0]
            is_link = isinstance(kind, str) and kind.isupper() and kind not in ("INT", "FLOAT", "STRING", "BOOLEAN")
            val = n["inputs"].get(name)
            if is_link:
                link = None
                if isinstance(val, list):
                    link_id += 1
                    link = link_id
                    links.append([link_id, int(val[0]), val[1], int(nid), len(inputs_json), kind])
                inputs_json.append({"name": name, "type": kind, "link": link})
            else:
                if val is None:
                    extra = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
                    val = extra.get("default", kind[0] if isinstance(kind, list) and kind else "")
                widgets.append(val)
                if name == "seed" or (isinstance(kind, str) and kind == "INT" and name.endswith("seed")):
                    widgets.append("fixed")  # control_after_generate
        outputs_json = []
        for i, (otype, oname) in enumerate(zip(spec["output"], spec["output_name"])):
            outputs_json.append({"name": oname, "type": otype, "links": [], "slot_index": i})
            out_slots[(nid, i)] = outputs_json[-1]["links"]
        y = col_y.get(depth[nid], 0)
        col_y[depth[nid]] = y + 180 + 24 * len(widgets)
        nodes.append({"id": int(nid), "type": ct, "pos": [depth[nid] * 420, y],
                      "size": [380, 120 + 24 * (len(inputs_json) + len(widgets))],
                      "flags": {}, "order": len(nodes), "mode": 0,
                      "inputs": inputs_json, "outputs": outputs_json,
                      "properties": {"Node name for S&R": ct}, "widgets_values": widgets})
    for lid, src, sidx, dst, didx, kind in links:
        out_slots[(str(src), sidx)].append(lid)
    return {"last_node_id": max(int(k) for k in api), "last_link_id": link_id,
            "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy", default=None, help="running ComfyUI URL, e.g. http://127.0.0.1:8188")
    a = ap.parse_args()
    api = build_api_graph()
    (HERE / "wan_animate_replace.api.json").write_text(json.dumps(api, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {HERE / 'wan_animate_replace.api.json'} ({len(api)} nodes)")
    cloud = build_cloud_graph()
    (HERE / "wan_animate_replace_cloud.api.json").write_text(json.dumps(cloud, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {HERE / 'wan_animate_replace_cloud.api.json'} ({len(cloud)} nodes)")
    if a.comfy:
        info = fetch_object_info(a.comfy)
        wf = to_editor_format(api, info)
        (HERE / "wan_animate_replace.json").write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {HERE / 'wan_animate_replace.json'} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
