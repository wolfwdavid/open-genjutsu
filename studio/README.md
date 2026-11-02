# Render Studio

A local web app over the recast pipeline: a dark editing-suite UI at `http://127.0.0.1:8765`
that reads and writes this repo's own folders (`work_ex/`, `refs/`). Nothing leaves the machine
except the optional handoff to `video_editor_ai`.

```
powershell -ExecutionPolicy Bypass -File studio\run_studio.ps1      # builds the UI on first run, opens the browser
```

| View | What it does | Reads / writes |
|---|---|---|
| **Monitor** | Live render progress: pass, windows done, s/frame, ETA, sampler step, Wan model load mode (partial = safe), ComfyUI / driver liveness, free RAM, OOM / device-lost / traceback alerts with the fix, log tail. Picks the newest `render_*.log`; older runs selectable. | `work_ex/round4/render_*.log`, `comfy_server.err.log` |
| **References** | Drop face close-ups (JPG/PNG/HEIC, EXIF-rotated on upload), build a candidate `user_ref` with `make_refs.py user --no-sunglasses-check`, see it next to the current one with the face-size verdict, promote it (old one kept as `user_ref_prev_*.png`). | `refs/face_uploads/`, `refs/user_ref_candidate*.png`, `refs/user_ref.png` |
| **Shot review** | The 5 camera shots on a strip; any two frame lanes side by side or with a wipe (source clip, cat pass, deflickered, pre-fix backup, shot re-renders, user pass); frame scrubber with timecodes; approve / re-roll + note per shot; video preview that jumps to the shot; **Send to editor**. | `work_ex/wan/**/frames*`, `work_ex/studio/review.json`, thumbnails cached in `work_ex/studio/cache/` |
| **Settings** | Editor URL, project-name prefix, the editor session cookie, connection test. | `studio/settings.json` (gitignored) |

Keyboard: `1`–`4` switch views; in Shot review `←` `→` step frames (Shift ×5), `A` approve, `R` re-roll, `W` wipe.

## Send to editor
`video_editor_ai` (FastAPI + HTMX, `<path to video_editor_ai>`, port 8000)
only accepts its own Supabase login session. The handoff therefore has two modes:

- **API** — with a saved `sb_access_token` cookie (Settings) the Studio creates a project
  (`POST /api/projects`) and uploads the video (`POST /api/projects/{id}/assets`), then links to it.
- **Manual** — if the editor is down or the session is missing/expired, the file is copied to
  `work_ex/handoff/` and the editor's projects page is opened; drag the staged file in.

Either way the chosen mp4 is staged first, so nothing is lost if the editor is unreachable.

## Development
```
cd studio/ui && npm run dev            # Vite on :5173 with /api proxied to :8765
.venv/Scripts/python.exe -m uvicorn studio.server:app --port 8765 --reload
```
Stack: FastAPI (in `.venv`) · Svelte 5 + Vite 8 + Tailwind 4 · Lucide icons · Fira Sans / Fira Code.
Design tokens: `studio/ui/src/app.css`; rationale: `studio/design-system/MASTER.md`.

Render launch buttons are deliberately absent: renders are started from the user's own PowerShell
(`run_example.ps1`), the Studio only watches them.
