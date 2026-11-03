#!/usr/bin/env python
"""
server.py -- POV Switch: a local web app that turns a clip into a branching first-person game.

  POST /api/clips            upload an mp4 -> analyze.py -> moments + characters
  POST /api/pov              {clip, moment, character, scene?} -> Qwen-Image-Edit POV frame (one deploy job)
  POST /api/outcome          {clip, pov, outcome} -> Wan2.2 I2V clip from that frame (one deploy job)
  GET  /api/clips/{id}       the clip's analysis and every branch generated so far (the game tree)
  GET  /files/{clip}/...     frames, crops, POV frames, outcome clips

Every generation is one `comfy deploy run` against the deployment in ../comfy-build.yaml.
Run:  .venv/Scripts/python.exe -m uvicorn game.server:app --host 127.0.0.1 --port 8766
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "game"))
import graphs  # noqa: E402

WORK = Path(os.environ.get("VG_WORK", ROOT / "work_ex")) / "game"
WORK.mkdir(parents=True, exist_ok=True)
SPEC = ROOT / "comfy-build.yaml"
PY = sys.executable
UI = ROOT / "game" / "ui"
LOCK = threading.Lock()

app = FastAPI(title="POV Switch")
app.mount("/files", StaticFiles(directory=str(WORK)), name="files")


def _clip_dir(cid: str) -> Path:
    d = WORK / cid
    if not d.is_dir() or not (d / "analysis.json").exists():
        raise HTTPException(404, "unknown clip")
    return d


def _state(cid: str) -> dict:
    d = _clip_dir(cid)
    a = json.load(open(d / "analysis.json", encoding="utf-8"))
    branches = json.load(open(d / "branches.json", encoding="utf-8")) if (d / "branches.json").exists() else []
    return dict(id=cid, analysis=a, branches=branches)


def _save_branch(cid: str, b: dict):
    d = _clip_dir(cid)
    with LOCK:
        bs = json.load(open(d / "branches.json", encoding="utf-8")) if (d / "branches.json").exists() else []
        bs.append(b)
        json.dump(bs, open(d / "branches.json", "w", encoding="utf-8"), indent=1)


def deploy_run(graph: dict, asset_root: Path, out_dir: Path, timeout: int = 1800) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    wf = out_dir / "workflow.api.json"
    wf.write_text(json.dumps(graph, indent=1, ensure_ascii=False), encoding="utf-8")
    cmd = ["comfy", "--json", "deploy", "run", str(SPEC), "--workflow", str(wf), "--asset-root", str(asset_root),
           "--output-dir", str(out_dir), "--timeout", str(timeout)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out_dir / "run.stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    try:
        env = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise HTTPException(502, f"comfy deploy run gave no envelope (exit {r.returncode}): {r.stdout[:500]}")
    (out_dir / "run.envelope.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    if not env.get("ok"):
        e = env.get("error") or {}
        job_err = ((e.get("details") or {}).get("job") or {}).get("error") or {}
        msg = job_err.get("message") or e.get("message") or e.get("code")
        raise HTTPException(502, f"{e.get('code')}: {msg}")
    d = env["data"]
    outs = [Path(o["path"]) for o in (d.get("outputs") or []) if o.get("path")]
    if not outs:
        outs = sorted(p for p in out_dir.rglob("*") if p.suffix.lower() in (".png", ".mp4", ".webp"))
    if not outs:
        raise HTTPException(502, "the job returned no output file")
    return dict(output=outs[0], seconds=round(time.time() - t0, 1), job=(d.get("job") or {}).get("id"))


@app.get("/", response_class=HTMLResponse)
def index():
    return (UI / "index.html").read_text(encoding="utf-8")


@app.post("/api/clips")
async def upload_clip(file: UploadFile = File(...)):
    cid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    d = WORK / cid
    d.mkdir(parents=True)
    src = d / "source.mp4"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    r = subprocess.run([PY, str(ROOT / "game" / "analyze.py"), "--video", str(src), "--out", str(d)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    (d / "analyze.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise HTTPException(500, "analyze failed: " + (r.stderr or r.stdout)[-800:])
    return _state(cid)


@app.get("/api/clips/{cid}")
def get_clip(cid: str):
    return _state(cid)


class PovReq(BaseModel):
    clip: str
    moment: str
    character: str
    name: str = ""          # who this is, in the player's words (required for the "someone else" card)
    scene: str = "in the middle of the action"
    seed: int = 0


@app.post("/api/pov")
def make_pov(req: PovReq):
    d = _clip_dir(req.clip)
    a = json.load(open(d / "analysis.json", encoding="utf-8"))
    m = next((m for m in a["moments"] if m["id"] == req.moment), None)
    if not m:
        raise HTTPException(404, "unknown moment")
    c = next((c for c in m["characters"] if c["id"] == req.character), None)
    if not c:
        raise HTTPException(404, "unknown character")
    others = ", ".join(x["label"] for x in m["characters"] if x["id"] != c["id"] and x["cls"] != "custom") or "nobody else"
    if c["cls"] == "custom":
        if not req.name.strip():
            raise HTTPException(400, "say who you are in this scene (e.g. 'the marine on the left')")
        who = req.name.strip()
    else:
        who = req.name.strip() or f"the {c['cls']}"
    label = who
    prompt = graphs.pov_prompt(who, req.scene, others)
    bid = "pov-" + uuid.uuid4().hex[:8]
    bdir = d / "branches" / bid
    bdir.mkdir(parents=True)
    # the POV is generated from the full source frame at this moment (1:1 with the clip), not the crop
    frame = d / m["frame"]
    seed = req.seed or int(time.time()) % 100000
    g = graphs.pov_frame(frame, prompt, a["width"], a["height"], seed, bid)
    try:
        res = deploy_run(g, d, bdir / "out")
    except HTTPException:
        shutil.rmtree(bdir, ignore_errors=True)
        raise
    pov = bdir / "pov.png"
    shutil.copy(res["output"], pov)
    b = dict(id=bid, kind="pov", moment=m["id"], character=c["id"], label=label, prompt=prompt, seed=seed,
             image=str(pov.relative_to(WORK)).replace("\\", "/"), seconds=res["seconds"], job=res["job"],
             parent=None)
    _save_branch(req.clip, b)
    return b


class OutcomeReq(BaseModel):
    clip: str
    pov: str
    outcome: str
    length: int = 49
    seed: int = 0


@app.post("/api/outcome")
def make_outcome(req: OutcomeReq):
    d = _clip_dir(req.clip)
    a = json.load(open(d / "analysis.json", encoding="utf-8"))
    bs = json.load(open(d / "branches.json", encoding="utf-8")) if (d / "branches.json").exists() else []
    p = next((b for b in bs if b["id"] == req.pov), None)
    if not p:
        raise HTTPException(404, "unknown POV branch")
    start = WORK / p["image"]
    bid = "out-" + uuid.uuid4().hex[:8]
    bdir = d / "branches" / bid
    bdir.mkdir(parents=True)
    # keep the source aspect; Wan likes ~0.4 MP
    w, h = a["width"], a["height"]
    scale = (832 * 480 / (w * h)) ** 0.5
    W, H = int(w * scale), int(h * scale)
    who = p["label"]
    prompt = graphs.outcome_prompt(who, req.outcome.strip().rstrip(".") + ".", "continuing the same scene")
    seed = req.seed or int(time.time()) % 100000
    length = max(9, min(81, (req.length // 4) * 4 + 1))
    g = graphs.outcome_clip(start, prompt, W, H, length, seed, bid)
    try:
        res = deploy_run(g, d, bdir / "out")
    except HTTPException:
        shutil.rmtree(bdir, ignore_errors=True)
        raise
    clip = bdir / "outcome.mp4"
    shutil.copy(res["output"], clip)
    # the last frame is the next decision point
    last = bdir / "last.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.1", "-i", str(clip), "-frames:v", "1", str(last)])
    b = dict(id=bid, kind="outcome", parent=p["id"], label=who, outcome=req.outcome, prompt=prompt, seed=seed,
             video=str(clip.relative_to(WORK)).replace("\\", "/"),
             image=str(last.relative_to(WORK)).replace("\\", "/") if last.exists() else None,
             seconds=res["seconds"], job=res["job"], width=W, height=H, length=length)
    _save_branch(req.clip, b)
    return b


class ContinueReq(BaseModel):
    clip: str
    branch: str
    outcome: str
    length: int = 49
    seed: int = 0


@app.post("/api/continue")
def continue_branch(req: ContinueReq):
    """Play on from an outcome's last frame: the same as /api/outcome but chained."""
    d = _clip_dir(req.clip)
    bs = json.load(open(d / "branches.json", encoding="utf-8"))
    p = next((b for b in bs if b["id"] == req.branch), None)
    if not p or not p.get("image"):
        raise HTTPException(404, "unknown branch")
    # reuse make_outcome with this branch acting as the start frame
    r = OutcomeReq(clip=req.clip, pov=p["id"], outcome=req.outcome, length=req.length, seed=req.seed)
    return make_outcome(r)


@app.get("/api/deployment")
def deployment():
    r = subprocess.run(["comfy", "--json", "deploy", "status", str(SPEC)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    try:
        env = json.loads(r.stdout)
        dep = (env.get("data") or {}).get("deployment") or {}
        return dict(status=dep.get("status"), endpoint=dep.get("endpointUrl"),
                    release=((env.get("data") or {}).get("release") or {}).get("version"))
    except json.JSONDecodeError:
        return dict(status="unknown")
