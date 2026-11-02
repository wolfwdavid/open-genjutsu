#!/usr/bin/env python
"""Render Studio backend: a local FastAPI app over the recast pipeline's files.

Serves three views (render monitor, reference photos, shot review) plus the
"Send to editor" handoff to video_editor_ai. Everything reads and writes the
repo's own folders (work_ex/, refs/); nothing leaves the machine except the
optional handoff to the editor URL in studio/settings.json.

    .venv/Scripts/python.exe -m uvicorn studio.server:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import psutil
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("VG_WORK", ROOT / "work_ex"))
if not WORK.is_absolute():
    WORK = ROOT / WORK
REFS = ROOT / "refs"
STUDIO_DIR = ROOT / "studio"
STATE_DIR = WORK / "studio"
CACHE_DIR = STATE_DIR / "cache"
UPLOAD_DIR = REFS / "face_uploads"
SETTINGS_PATH = STUDIO_DIR / "settings.json"
REVIEW_PATH = STATE_DIR / "review.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
FPS = 24
STRIDE = 2

# shot boundaries in strided frames (the render's frame space), from run_example.ps1
SHOTS = [(0, 52), (52, 92), (92, 138), (138, 173), (173, 181)]
DEFAULT_SPF = 28.4  # s/frame measured on the Arc 140T at 768x576

app = FastAPI(title="Render Studio", docs_url=None, redoc_url=None)
for d in (STATE_DIR, CACHE_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- helpers
def read_text(path: Path, max_bytes: int = 4_000_000) -> str:
    """Logs written through Tee-Object mix UTF-16 and UTF-8: drop NULs, ignore bad bytes."""
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        if path.stat().st_size > max_bytes:
            f.seek(-max_bytes, 2)
        raw = f.read()
    return raw.replace(b"\x00", b"").decode("utf-8", errors="ignore")


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def safe_path(rel: str) -> Path:
    """Resolve a repo-relative path and refuse anything outside refs/ or work_ex/."""
    p = (ROOT / rel).resolve()
    for base in (REFS.resolve(), WORK.resolve()):
        if p == base or base in p.parents:
            return p
    raise HTTPException(403, "path outside refs/ or work_ex/")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def fmt_hms(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    s = max(0, int(seconds))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ----------------------------------------------------------------------------- render monitor
RE_PLAN = re.compile(r"\[(\d\d:\d\d:\d\d)\] (\d+) frames -> (\d+) window\(s\): (.*)")
RE_PLAN_WIN = re.compile(r"w(\d+)\[seq (\d+)\+(\d+), new (\d+)\]")
RE_WIN_START = re.compile(r"\[(\d\d:\d\d:\d\d)\] window (\d+): seq (\d+)\.\.(\d+) \((\d+) frames, (\d+)x(\d+), steps (\d+)\)")
RE_WIN_DONE = re.compile(r"\[(\d\d:\d\d:\d\d)\] window (\d+): (\d+) frames in (\d+) s = ([\d.]+) s/frame")
RE_WIN_SKIP = re.compile(r"\[(\d\d:\d\d:\d\d)\] window (\d+): already rendered \((\d+) frames\)")
RE_TOTAL = re.compile(r"\[(\d\d:\d\d:\d\d)\] total (\d+) frames in ([\d.]+) min")
RE_STEP = re.compile(r"(\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+),\s*([\d.]+)s/it\]")
RE_LOADED = re.compile(r"loaded (partially|completely); ([\d.]+) MB usable, ([\d.]+) MB loaded(?:, ([\d.]+) MB offloaded)?")


def log_time_to_epoch(hhmmss: str, mtime: float) -> float:
    """Log lines carry only a clock time; anchor them to the log file's date."""
    day = datetime.fromtimestamp(mtime).replace(hour=0, minute=0, second=0, microsecond=0)
    h, m, s = (int(x) for x in hhmmss.split(":"))
    t = day.timestamp() + h * 3600 + m * 60 + s
    if t > mtime + 60:  # the line is from the previous day
        t -= 86400
    return t


def run_meta(log_path: Path) -> dict:
    m = re.match(r"render_(cat|user)_(full|probe|fix(\d+))\.log", log_path.name)
    side, tag, shot = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", None)
    if shot:
        name, label = f"cat_E_shot{shot}", f"Cat · re-render shot {shot}"
    elif tag == "probe":
        name, label = f"{side}_E_probe", f"{'Cat' if side == 'cat' else 'User'} · probe (first 33 frames)"
    else:
        name, label = f"{side}_E", f"{'Cat' if side == 'cat' else 'User'} · full pass"
    return dict(side=side, tag=tag, shot=int(shot) if shot else None, name=name, label=label)


def parse_render_log(log_path: Path) -> dict:
    text = read_text(log_path)
    mtime = log_path.stat().st_mtime
    # only the latest attempt: everything after the last "staged" line
    idx = text.rfind("] staged ")
    seg = text[text.rfind("\n", 0, idx) + 1:] if idx >= 0 else text
    lines = seg.splitlines()

    plan, windows, started_at = None, {}, None
    for ln in lines:
        if (mp := RE_PLAN.search(ln)):
            started_at = log_time_to_epoch(mp.group(1), mtime)
            plan = dict(frames=int(mp.group(2)), n_windows=int(mp.group(3)))
            for w in RE_PLAN_WIN.finditer(mp.group(4)):
                windows[int(w.group(1))] = dict(index=int(w.group(1)), seq=int(w.group(2)), length=int(w.group(3)),
                                                new=int(w.group(4)), state="pending", spf=None, seconds=None)
        elif (ms := RE_WIN_START.search(ln)):
            i = int(ms.group(2))
            windows.setdefault(i, dict(index=i, seq=int(ms.group(3)), length=int(ms.group(5)), new=int(ms.group(5)),
                                       state="pending", spf=None, seconds=None))
            windows[i].update(state="running", started=log_time_to_epoch(ms.group(1), mtime),
                              size=f"{ms.group(6)}x{ms.group(7)}", steps=int(ms.group(8)))
        elif (md := RE_WIN_DONE.search(ln)):
            i = int(md.group(2))
            windows.setdefault(i, dict(index=i, seq=0, length=0, new=int(md.group(3)), state="pending", spf=None, seconds=None))
            windows[i].update(state="done", seconds=int(md.group(4)), spf=float(md.group(5)))
        elif (mk := RE_WIN_SKIP.search(ln)):
            i = int(mk.group(2))
            windows.setdefault(i, dict(index=i, seq=0, length=0, new=int(mk.group(3)), state="pending", spf=None, seconds=None))
            windows[i].update(state="skipped")

    alerts = []
    if "XPU out of memory" in seg or "OutOfMemoryError" in seg:
        alerts.append(dict(level="error", code="oom", text="GPU out of memory in KSampler.",
                           action="Close GPU-heavy apps (browser video, editors) and re-run the same command; "
                                  "it resumes. run_example.ps1 reserves 4 GB (-ReserveVram); raise it to 5 if it repeats."))
    if "device lost" in seg.lower():
        alerts.append(dict(level="error", code="device_lost", text="GPU device lost (driver reset).",
                           action="Re-run the same command; finished windows are kept."))
    if "Traceback" in seg and not alerts:
        alerts.append(dict(level="error", code="traceback", text="The render script raised an exception.",
                           action="See the log tail below."))
    failed = any(a["level"] == "error" for a in alerts) or "failed (exit" in seg
    done = "ALL DONE" in seg
    total = RE_TOTAL.search(seg)

    ws = [windows[k] for k in sorted(windows)]
    done_frames = sum(w["new"] for w in ws if w["state"] in ("done", "skipped"))
    # a probe plans all windows but renders one: once ALL DONE, what was rendered is the whole job
    planned = done_frames if done else plan["frames"] if plan else sum(w["new"] for w in ws)
    current = next((w for w in ws if w["state"] == "running"), None)
    # frame-weighted s/frame (a plain mean lets a 7-frame window at 52 s/frame skew the ETA)
    timed = [w for w in ws if w["spf"] and w["seconds"]]
    spf = sum(w["seconds"] for w in timed) / max(1, sum(w["new"] for w in timed)) if timed else None

    state = "done" if done else "failed" if failed else "running" if current else "idle"
    meta = run_meta(log_path)
    frames_dir = WORK / "wan" / meta["name"] / "frames"
    on_disk = len(glob.glob(str(frames_dir / "*.png"))) if frames_dir.exists() else 0
    return dict(log=str(log_path.relative_to(ROOT)).replace("\\", "/"), mtime=mtime, started_at=started_at,
                state=state, plan=plan, windows=ws, planned_frames=planned, done_frames=done_frames,
                current=current, spf=spf, alerts=alerts, total_line=total.group(0) if total else None,
                frames_on_disk=on_disk, frames_dir=str(frames_dir.relative_to(ROOT)).replace("\\", "/"),
                tail=[ln for ln in lines if "... waiting" not in ln][-40:], **meta)


def parse_comfy_log() -> dict:
    path = WORK / "round4" / "comfy_server.err.log"
    text = ANSI.sub("", read_text(path, 1_500_000)).replace("\r", "\n")
    load, step = None, None
    pos = text.rfind("Requested to load WAN22_Animate")
    if pos >= 0 and (ml := RE_LOADED.search(text, pos)):
        load = dict(mode=ml.group(1), usable_mb=float(ml.group(2)), loaded_mb=float(ml.group(3)),
                    offloaded_mb=float(ml.group(4)) if ml.group(4) else 0.0)
    steps = RE_STEP.findall(text)
    if steps:
        k, n, el, eta, sit = steps[-1]
        step = dict(k=int(k), n=int(n), elapsed=el, eta=eta, s_per_it=float(sit))
    return dict(exists=path.exists(), mtime=path.stat().st_mtime if path.exists() else None, load=load, step=step,
                oom_count=text.count("Got an OOM"))


_proc_cache: dict = {"t": 0.0, "v": None}


def processes() -> dict:
    """ComfyUI server / render driver liveness, cached for 5 s (cmdline scans are slow on Windows)."""
    now = time.time()
    if _proc_cache["v"] is not None and now - _proc_cache["t"] < 5:
        return _proc_cache["v"]
    comfy = driver = False
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "ComfyUI" in cmd and "main.py" in cmd:
            comfy = True
        if "wan_animate.py" in cmd:
            driver = True
    vm = psutil.virtual_memory()
    v = dict(comfy=comfy, driver=driver, ram_free_gb=round(vm.available / 2**30, 1), ram_total_gb=round(vm.total / 2**30, 1))
    _proc_cache.update(t=now, v=v)
    return v


@app.get("/api/status")
def api_status(log: str | None = None):
    logs = sorted((WORK / "round4").glob("render_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if log:
        chosen = next((p for p in logs if p.name == log), None)
        if chosen is None:
            raise HTTPException(404, "log not found")
    else:
        chosen = logs[0] if logs else None
    run = parse_render_log(chosen) if chosen else None
    comfy = parse_comfy_log()
    procs = processes()

    if run:
        # a run whose log stopped updating and whose driver is gone is not running any more
        if run["state"] == "running" and not procs["driver"] and time.time() - run["mtime"] > 180:
            run["state"] = "stalled"
        # progress incl. the in-flight window (fraction of sampler steps)
        partial = 0.0
        if run["current"] and comfy["step"] and comfy["mtime"] and comfy["mtime"] >= run["mtime"] - 600:
            partial = run["current"]["new"] * comfy["step"]["k"] / max(1, comfy["step"]["n"])
        done = run["done_frames"] + partial
        spf = run["spf"] or DEFAULT_SPF
        remaining = max(0.0, run["planned_frames"] - done)
        eta = remaining * spf if run["state"] == "running" else None
        run.update(progress=min(1.0, done / run["planned_frames"]) if run["planned_frames"] else 0.0,
                   eta_seconds=eta, eta_text=fmt_hms(eta),
                   eta_clock=datetime.fromtimestamp(time.time() + eta).strftime("%H:%M") if eta else None,
                   elapsed_text=fmt_hms(time.time() - run["started_at"]) if run["started_at"] and run["state"] == "running" else None)
        if procs["ram_free_gb"] < 2 and run["state"] == "running":
            run["alerts"].append(dict(level="warn", code="ram", text=f"Only {procs['ram_free_gb']} GB of RAM free.",
                                      action="Close Chrome / VS Code while rendering; the offloaded model weights live in RAM."))
        if comfy["load"] and comfy["load"]["mode"] == "completely" and comfy["oom_count"] and run["state"] != "done":
            run["alerts"].append(dict(level="warn", code="full_load", text="ComfyUI loaded the Wan model completely and hit OOM.",
                                      action="Re-run with a larger reserve: run_example.ps1 ... -ReserveVram 5 (forces a partial load)."))
    return dict(run=run, comfy=comfy, procs=procs, now=time.time(),
                logs=[dict(file=p.name, mtime=p.stat().st_mtime, **run_meta(p)) for p in logs])


# ----------------------------------------------------------------------------- jobs (subprocesses)
class Job:
    def __init__(self, kind: str, cmd: list[str]):
        self.id = uuid.uuid4().hex[:8]
        self.kind, self.cmd = kind, cmd
        self.state, self.rc = "running", None
        self.lines: deque[str] = deque(maxlen=400)
        self.started, self.ended = time.time(), None
        self.result: dict = {}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        env = dict(os.environ, VG_WORK=str(WORK), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            p = subprocess.Popen(self.cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for ln in p.stdout:
                self.lines.append(ln.rstrip())
            self.rc = p.wait()
        except Exception as e:  # noqa: BLE001
            self.lines.append(f"studio: {e}")
            self.rc = -1
        self.ended = time.time()
        self.state = "done" if self.rc == 0 else "failed"

    def to_dict(self):
        return dict(id=self.id, kind=self.kind, state=self.state, rc=self.rc, started=self.started, ended=self.ended,
                    lines=list(self.lines), result=self.result, cmd=" ".join(self.cmd))


JOBS: dict[str, Job] = {}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    d = job.to_dict()
    if job.kind == "build_ref" and job.state == "done":
        d["result"] = ref_build_result(job)
    return d


# ----------------------------------------------------------------------------- reference photos
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


def image_info(path: Path) -> dict | None:
    if not path.exists():
        return None
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return dict(path=str(path.relative_to(ROOT)).replace("\\", "/"), width=w, height=h, bytes=path.stat().st_size,
                mtime=path.stat().st_mtime)


@app.get("/api/refs")
def api_refs():
    uploads = sorted(UPLOAD_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    return dict(current=image_info(REFS / "user_ref.png"), candidate=image_info(REFS / "user_ref_candidate.png"),
                candidate_sheet=image_info(REFS / "user_ref_candidate_sheet.jpg"),
                previous=[image_info(p) for p in sorted(REFS.glob("user_ref_prev_*.png"), reverse=True)[:5]],
                uploads=[image_info(p) for p in uploads])


@app.post("/api/refs/upload")
async def api_refs_upload(files: list[UploadFile] = File(...)):
    from PIL import Image, ImageOps
    saved = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise HTTPException(415, f"{f.filename}: not an image ({suffix or 'no extension'})")
        if suffix in (".heic", ".heif"):
            import pillow_heif
            pillow_heif.register_heif_opener()
        data = await f.read()
        tmp = UPLOAD_DIR / f"_incoming{suffix}"
        tmp.write_bytes(data)
        try:
            with Image.open(tmp) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")   # phone photos are stored sideways + EXIF tag
                im.thumbnail((3000, 3000))
                name = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}_{Path(f.filename).stem[:40]}.jpg"
                out = UPLOAD_DIR / name
                im.save(out, "JPEG", quality=95)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"{f.filename}: cannot decode ({e})")
        finally:
            tmp.unlink(missing_ok=True)
        saved.append(image_info(out))
    return dict(saved=saved)


@app.delete("/api/refs/upload/{name}")
def api_refs_delete(name: str):
    p = UPLOAD_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404, "no such upload")
    p.unlink()
    return dict(ok=True)


@app.post("/api/refs/build")
def api_refs_build():
    if not list(UPLOAD_DIR.glob("*.jpg")):
        raise HTTPException(400, "upload at least one photo first")
    if any(j.kind == "build_ref" and j.state == "running" for j in JOBS.values()):
        raise HTTPException(409, "a reference build is already running")
    for stale in (REFS / "user_ref_candidate.png", REFS / "user_ref_candidate_sheet.jpg"):
        stale.unlink(missing_ok=True)
    # no sunglasses check: the render prompt adds the black sunglasses, the photos are bare-faced
    cmd = [str(PY), "make_refs.py", "user", "--photos", "refs/face_uploads/*.jpg",
           "--out", "refs/user_ref_candidate.png", "--no-sunglasses-check"]
    job = Job("build_ref", cmd)
    JOBS[job.id] = job
    return dict(job=job.id)


RE_REF_ROW = re.compile(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([-+]?[\d.]+)\s*$")


def ref_build_result(job: Job) -> dict:
    """The best photo's row of make_refs' ranking table (first row after the header)."""
    rows = []
    for ln in job.lines:
        if (m := RE_REF_ROW.match(ln.strip())):
            rows.append(dict(photo=m.group(1), score=float(m.group(2)), sharp=float(m.group(8)), face_h=int(m.group(9)),
                             yaw=float(m.group(10)), body_ratio=float(m.group(5))))
    return dict(ranking=rows, best=rows[0] if rows else None, candidate=image_info(REFS / "user_ref_candidate.png"),
                sheet=image_info(REFS / "user_ref_candidate_sheet.jpg"))


@app.post("/api/refs/promote")
def api_refs_promote():
    cand = REFS / "user_ref_candidate.png"
    if not cand.exists():
        raise HTTPException(400, "no candidate reference; build one first")
    cur = REFS / "user_ref.png"
    backup = None
    if cur.exists():
        backup = REFS / f"user_ref_prev_{datetime.now():%Y%m%d_%H%M%S}.png"
        shutil.copy2(cur, backup)
    shutil.copy2(cand, cur)
    return dict(current=image_info(cur), backup=image_info(backup) if backup else None)


# ----------------------------------------------------------------------------- files, frames, videos
@app.get("/api/file")
def api_file(path: str):
    p = safe_path(path)
    if not p.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(p, headers={"Cache-Control": "no-cache"})


def lanes() -> dict[str, dict]:
    """Frame lanes in strided frame space (index i = cut frame 2i). offset = file index of strided frame 0."""
    wan = WORK / "wan"
    user_dir = wan / "user_E" / "frames"
    user_label = "User pass"
    if not user_dir.exists() and (wan / "user_E_oldref" / "frames").exists():
        user_dir, user_label = wan / "user_E_oldref" / "frames", "User pass (old reference)"
    cands = {
        "bg_masked": dict(label="Staged background (cat masked)", dir=wan / "cat_E" / "in" / "bg", offset=0),
        "cat": dict(label="Cat pass", dir=wan / "cat_E" / "frames", offset=0),
        "cat_df": dict(label="Cat pass · deflickered", dir=wan / "cat_E" / "frames_df", offset=0),
        "cat_backup": dict(label="Cat pass · before shot fixes", dir=wan / "cat_E" / "frames_backup", offset=0),
        "user": dict(label=user_label, dir=user_dir, offset=0),
        "user_df": dict(label="User pass · deflickered", dir=user_dir.parent / "frames_df", offset=0),
    }
    for p in sorted(wan.glob("cat_E_shot*")):
        if (p / "frames").exists() and (m := re.search(r"shot(\d+)$", p.name)):
            n = int(m.group(1))
            cands[f"cat_shot{n}"] = dict(label=f"Cat · shot {n} re-render", dir=p / "frames", offset=-SHOTS[n - 1][0])
    out = {}
    cut = source_clip()
    if cut:
        # the real source, decoded on demand (cut frame 2i of the 24 fps clip) and cached like the PNG lanes
        out["source"] = dict(key="source", label="Source clip", dir=str(cut.relative_to(ROOT)).replace("\\", "/"),
                             offset=0, count=SHOTS[-1][1], first=0, last=SHOTS[-1][1] - 1, video=True)
    for key, c in cands.items():
        if c["dir"].exists():
            count = len(glob.glob(str(c["dir"] / "*.png")))
            lo = -c["offset"]
            out[key] = dict(key=key, label=c["label"], dir=str(c["dir"].relative_to(ROOT)).replace("\\", "/"),
                            offset=c["offset"], count=count, first=lo, last=lo + count - 1, video=False)
    return out


def source_clip() -> Path | None:
    blocks = load_json(WORK / "blocks.json", {})
    p = ROOT / blocks.get("cut", "") if blocks.get("cut") else WORK / "cut.mp4"
    return p if p.is_file() else None


def lane_file(lane: dict, i: int) -> Path:
    return ROOT / lane["dir"] / f"{i + lane['offset']:05d}.png"


def extract_source_frame(cut: Path, i: int, out: Path) -> None:
    """Decode cut frame 2i (the strided index) to a PNG with ffmpeg. Input seeking (-ss before -i) is
    frame-accurate on mp4 and ~0.2 s; a select filter from the start took 8 s per frame. -ss returns the
    first frame with pts >= the seek point, so aim half a frame before the target."""
    frame = i * STRIDE
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, frame - 0.5) / FPS:.5f}", "-i", str(cut), "-frames:v", "1", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or not out.exists():
        raise HTTPException(500, f"ffmpeg failed: {r.stderr.strip()[:200]}")


@app.get("/api/shots")
def api_shots():
    review = load_json(REVIEW_PATH, {})
    ls = lanes()
    shots = []
    for n, (a, b) in enumerate(SHOTS, start=1):
        avail = [k for k, l in ls.items() if l["first"] <= a and l["last"] >= b - 1]
        shots.append(dict(n=n, start=a, end=b, frames=b - a, cut_start=a * STRIDE, cut_end=b * STRIDE,
                          t_start=round(a * STRIDE / FPS, 3), t_end=round(b * STRIDE / FPS, 3),
                          lanes=avail, review=review.get(str(n))))
    return dict(shots=shots, lanes=ls, fps=FPS, stride=STRIDE, total=SHOTS[-1][1])


class ReviewIn(BaseModel):
    decision: str | None = None   # approve | reroll | None (clear)
    note: str = ""


@app.post("/api/shots/{n}/review")
def api_review(n: int, body: ReviewIn):
    if not 1 <= n <= len(SHOTS):
        raise HTTPException(404, "no such shot")
    if body.decision not in (None, "approve", "reroll"):
        raise HTTPException(422, "decision must be approve, reroll or null")
    review = load_json(REVIEW_PATH, {})
    if body.decision is None and not body.note.strip():
        review.pop(str(n), None)
    else:
        review[str(n)] = dict(decision=body.decision, note=body.note.strip(), at=time.time())
    save_json(REVIEW_PATH, review)
    return dict(review=review.get(str(n)))


@app.get("/api/frame/{lane}/{i}")
def api_frame(lane: str, i: int, w: int = Query(480, ge=64, le=1920)):
    ls = lanes()
    if lane not in ls:
        raise HTTPException(404, "no such lane")
    if ls[lane].get("video"):
        if not 0 <= i < ls[lane]["count"]:
            raise HTTPException(404, "no such frame")
        cut = ROOT / ls[lane]["dir"]
        src = CACHE_DIR / f"source_{i:05d}_{int(cut.stat().st_mtime)}.png"
        if not src.exists():
            extract_source_frame(cut, i, src)
    else:
        src = lane_file(ls[lane], i)
        if not src.exists():
            raise HTTPException(404, "no such frame")
    key = f"{lane}_{i:05d}_{w}_{int(src.stat().st_mtime)}.jpg"
    out = CACHE_DIR / key
    if not out.exists():
        from PIL import Image
        with Image.open(src) as im:
            im = im.convert("RGB")
            if im.width > w:
                im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
            im.save(out, "JPEG", quality=86)
    # revalidate every time: frames_df / spliced frames are rewritten in place after shot fixes, the URL never changes
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@app.get("/api/videos")
def api_videos():
    vids = []
    for p in sorted(list(WORK.glob("*.mp4")) + list((WORK / "wan").glob("*.mp4")), key=lambda p: p.stat().st_mtime, reverse=True):
        vids.append(dict(name=p.name, path=str(p.relative_to(ROOT)).replace("\\", "/"), bytes=p.stat().st_size, mtime=p.stat().st_mtime))
    return dict(videos=vids)


# ----------------------------------------------------------------------------- handoff to video_editor_ai
DEFAULT_SETTINGS = dict(editor_url="http://127.0.0.1:8000", editor_cookie="", project_prefix="Cat duet")


def settings() -> dict:
    return {**DEFAULT_SETTINGS, **load_json(SETTINGS_PATH, {})}


class SettingsIn(BaseModel):
    editor_url: str | None = None
    editor_cookie: str | None = None
    project_prefix: str | None = None


@app.get("/api/settings")
def api_settings():
    s = settings()
    return dict(editor_url=s["editor_url"], project_prefix=s["project_prefix"], has_cookie=bool(s["editor_cookie"]))


@app.post("/api/settings")
def api_settings_save(body: SettingsIn):
    s = settings()
    for k, v in body.model_dump(exclude_none=True).items():
        s[k] = v.strip() if isinstance(v, str) else v
    s["editor_url"] = s["editor_url"].rstrip("/")
    save_json(SETTINGS_PATH, s)
    return api_settings()


@app.get("/api/handoff/ping")
def api_handoff_ping():
    import httpx
    s = settings()
    try:
        r = httpx.get(s["editor_url"] + "/", timeout=3, follow_redirects=False)
        reachable = True
        session_ok = None
        if s["editor_cookie"]:
            r2 = httpx.get(s["editor_url"] + "/projects", cookies={"sb_access_token": s["editor_cookie"]}, timeout=5,
                           follow_redirects=False)
            session_ok = r2.status_code == 200
        return dict(reachable=reachable, status=r.status_code, session_ok=session_ok, url=s["editor_url"])
    except httpx.HTTPError as e:
        return dict(reachable=False, error=str(e), session_ok=None, url=s["editor_url"])


class HandoffIn(BaseModel):
    video: str                 # repo-relative path under work_ex/
    name: str | None = None


@app.post("/api/handoff")
def api_handoff(body: HandoffIn):
    import httpx
    src = safe_path(body.video)
    if not src.is_file() or src.suffix.lower() != ".mp4":
        raise HTTPException(404, "video not found")
    s = settings()
    name = (body.name or f"{s['project_prefix']} {datetime.now():%Y-%m-%d %H:%M}").strip()
    stage_dir = WORK / "handoff"
    stage_dir.mkdir(exist_ok=True)
    staged = stage_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{src.name}"
    shutil.copy2(src, staged)
    base = dict(staged=str(staged), staged_rel=str(staged.relative_to(ROOT)).replace("\\", "/"),
                editor_url=s["editor_url"], projects_url=s["editor_url"] + "/projects", name=name)
    if not s["editor_cookie"]:
        return dict(mode="manual", reason="No editor session saved in Settings: open the editor and upload the staged file.", **base)
    cookies = {"sb_access_token": s["editor_cookie"]}
    try:
        r = httpx.post(s["editor_url"] + "/api/projects", data={"name": name}, cookies=cookies, timeout=15, follow_redirects=False)
        if r.status_code == 401:
            return dict(mode="manual", reason="The editor rejected the saved session (expired). Sign in again and paste a fresh cookie.", **base)
        loc = r.headers.get("location", "")
        m = re.search(r"/projects/([^/?#]+)", loc)
        if r.status_code not in (302, 303) or not m:
            return dict(mode="manual", reason=f"Unexpected reply creating the project ({r.status_code}).", **base)
        pid = m.group(1)
        with open(staged, "rb") as fh:
            r2 = httpx.post(f"{s['editor_url']}/api/projects/{pid}/assets", files={"video": (src.name, fh, "video/mp4")},
                            cookies=cookies, timeout=300)
        if r2.status_code >= 400:
            return dict(mode="manual", reason=f"Project created but the upload failed ({r2.status_code}): {r2.text[:200]}",
                        project_url=f"{s['editor_url']}/projects/{pid}", **base)
        return dict(mode="api", project_id=pid, project_url=f"{s['editor_url']}/projects/{pid}", **base)
    except httpx.HTTPError as e:
        return dict(mode="manual", reason=f"Editor not reachable at {s['editor_url']} ({e.__class__.__name__}). Start it with python run_app.py.", **base)


# ----------------------------------------------------------------------------- static UI
DIST = STUDIO_DIR / "ui" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        f = DIST / full_path
        if full_path and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def no_ui():
        return Response("Render Studio UI is not built. Run studio/run_studio.ps1 (or: cd studio/ui && npm run build).",
                        media_type="text/plain")
