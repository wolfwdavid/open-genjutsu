# Lessons (this build)

- A `grep '"version": 2'` on `comfy deploy status` is a false positive: `latestDeployable.version` carries the new
  number while the deployment still serves the old release. Parse `release.version` and `release.behind`.
- `uv run` picks up the nearest pyproject: running it from inside a junction that points into another project
  (work_ex -> synaps-optic-context) synced that project's venv. Run `uv run` from the repo root only.
- `comfy build init` only collects .ckpt/.pt/.bin/.pth/.safetensors; GGUF and ONNX weights must be added by hand
  (`comfy build refs resolve <filename>` gives the public source + digest).
- A bare package name in `pipDependencies` installs nothing; pin it (`matplotlib>=3.8`).
- Validate a ComfyUI graph without a GPU: zero-byte placeholder files make the loader combos accept the names,
  POST to /prompt, then interrupt + delete the queue entry before it executes.
- Long unattended sessions on a laptop: Windows sleeps during a 400 s `pip install` and the clock jumps 16 h on
  wake. Check `date -u` against the last known server timestamp before trusting "how much time is left".
- Comfy platform credits belong to the workspace, not the user: a shared team workspace can go negative
  overnight from other members' usage and stops every deployment (`stopReason: credits`). Run
  `comfy cloud status` before planning platform work.
- A custom node does not need a public git repo for a build: `customNodes: - source: local, localPath: <dir>`
  zips the folder from the custom-nodes dir (symlinks skipped) and uploads it; its requirements.txt is installed.
- Matrix-Game 2.0 on workers without flash-attn: only `action_module.py` and the i2v cross-attention call
  flash-attn unconditionally; an SDPA `flash_attn_func` with the same [B, L, H, D] layout is enough, and the
  KV-cache path never reaches flex_attention. Rotary tables assume CUDA at construction; patch for CPU tests.
- The Claude Code auto-mode classifier blocks `gh repo create` (new public surface); local commits are fine.
