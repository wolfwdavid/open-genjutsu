# POV Switch

Turn any clip into a playable, branching first-person game. Drop a video, pick anyone in it, see
the moment through their eyes, choose how it ends, keep going.

Inspired by the "Halo Wars, but first person" meme: the same battle, seen from inside it. Here it
works on any footage, including the cat.

## How to play

1. **Drop a clip.** The analyzer samples frames, finds shot cuts (HSV histogram correlation) and
   the peaks of motion inside each shot, and runs YOLOv10m on the candidate peaks. The moments of
   decision are the peaks with someone in them.
2. **Pick a moment, then whose eyes.** Every detected person or animal is a card with a 1:1 crop.
   There is always a "someone else here" card for things the detector cannot name (a marine in a
   game render, a car): you type who you are.
3. **See it from their eyes.** Qwen-Image-Edit-2509 rewrites the exact source frame as that
   character's first-person view, same place, same moment, everyone else where they were.
4. **Choose the outcome.** "I win", "I lose", "draw", "run", or your own words. Wan2.2 I2V plays
   the next three seconds forward from the POV frame with that outcome.
5. **Keep going.** The last frame of every outcome is a new decision point, so the clip becomes a
   tree of branches, all shown under the stage.

## What runs where

| step | model | where |
|---|---|---|
| frames, cuts, peaks | ffmpeg + OpenCV | laptop CPU, seconds |
| characters | YOLOv10m ONNX (COCO) | laptop CPU |
| POV frame | Qwen-Image-Edit-2509 GGUF Q4 + Lightning 4-step LoRA | Comfy platform deployment, one `comfy deploy run` |
| outcome clip | Wan2.2 I2V A14B GGUF Q4, high + low experts, lightx2v 4-step LoRAs | same deployment, one `comfy deploy run` |

All weights are open and resolved by sha256 to public Hugging Face sources in `comfy-build.yaml`
(release v2 of the build). Both graphs are in `game/graphs.py` and are plain API-format prompts;
nothing custom beyond ComfyUI-GGUF and VideoHelperSuite.

## Run it

```
.venv/Scripts/python.exe -m uvicorn game.server:app --host 127.0.0.1 --port 8766
```

Needs the pipeline venv (section 1 of `docs/SETUP.md`), ffmpeg, `comfy cloud login`, and the
deployment in `ready` state on release v2 (`comfy deploy status comfy-build.yaml`). The analyzer
alone runs with no deployment:

```
python game/analyze.py --video clip.mp4 --out work_ex/game/my-clip
```

## Files

| path | role |
|---|---|
| `game/analyze.py` | clip -> `analysis.json` (moments, characters, crops) |
| `game/graphs.py` | the two ComfyUI graphs + the prompt templates |
| `game/server.py` | FastAPI: upload, POV, outcome, continue, game tree |
| `game/ui/index.html` | the game screen |
| `game/PLAN.md` | the plan and the assumptions made without asking |
