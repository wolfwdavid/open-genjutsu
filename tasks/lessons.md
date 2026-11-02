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
