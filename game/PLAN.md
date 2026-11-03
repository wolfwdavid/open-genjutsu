# POV Switch: turn any clip into a playable branching game

Second app of the open-genjutsu entry. Plan written 2026-11-02 evening, deadline 2026-11-03 08:30 PT.

**What the player does.** Drop a clip. The app finds the characters (people and animals, YOLOv10)
and the moments of decision (peaks of motion, shot cuts). Pick a character: the view becomes that
character's first-person point of view at that moment, 1:1 with the source frame. Pick an outcome
(win, lose, or your own words). The model plays out the next seconds from those eyes with that
outcome. Every result is a branch you can keep playing from, so the clip becomes a tree.

**Open weights, on the Comfy developer platform.**
- POV frame: Qwen-Image-Edit-2509 (GGUF Q4) with the 4-step Lightning LoRA, prompted with
  "first-person view from <character>'s eyes".
- Outcome clip: Wan2.2 I2V A14B (GGUF Q4, high + low noise experts) with the 4-step Lightning LoRAs,
  started from the POV frame.
- Detection: YOLOv10m ONNX (already in the build) runs locally on CPU.
- Both graphs are one `comfy deploy run` each against the existing deployment, after the build is
  re-cut with the new weights (release v2).

**Pieces**
- [ ] `game/analyze.py`: frames, YOLO characters, decision moments, 1:1 crops -> `analysis.json`
- [ ] `game/graphs.py`: API-format graphs for the POV edit and the I2V outcome
- [ ] `game/server.py` + `game/ui/index.html`: upload, pick character, pick outcome, play branch
- [ ] Build spec: add 7 weights, push, release v2, move the deployment, test one POV + one outcome job
- [ ] README section, demo frames, commit

**Assumptions made without asking** (the user chose to skip questions)
- "1 to 1 ratio" = the POV frame is generated from the exact source frame at the chosen moment and
  the clip keeps the source aspect; a square 1:1 crop is an option, not the default.
- Outcome presets: "I win", "I lose", "nobody wins", plus free text.
- Clip length 49 frames at 16 fps (3 s) at 832x480 (or 480x832 for portrait sources).

## Lane 3 (2026-11-03): walk it live with Matrix-Game 2.0

The user asked for an "AAA" real-time feel. Matrix-Game 2.0 (Skywork, MIT, 1.8B, 352x640, 3 denoising
steps, keyboard + camera conditioning, KV-cache streaming) is wrapped as `ComfyUI-MatrixGame2` (sibling
repo, shipped into the build as a local pack). `/api/walk` sends the branch's frame plus a list of actions;
the node keeps the world (KV caches, decoder cache) resident on the worker keyed by the branch id, so the
next key press continues the same world in one short job instead of re-imagining the scene. Status: built,
CPU-validated, release v3 green; not yet run on a GPU because the workspace ran out of credit.
