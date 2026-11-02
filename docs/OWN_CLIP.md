# Reproducing the demo, then recasting your own clip

All work files live under one folder chosen by the `VG_WORK` environment variable
(`run_example.ps1` sets it to `work_ex/`). Nothing under it is tracked.

## A. Reproduce the demo

1. Put the source clip at `work_ex/cut.mp4` and its audio at `work_ex/audio.m4a`
   (`ffmpeg -i cut.mp4 -vn -c:a copy audio.m4a`). The demo used Higgsfield's public Genjutsu
   example clip (15 s, 960x720, 24 fps); it is not redistributed here.
2. Per-frame inputs (a few minutes on CPU/DirectML):
   ```
   set VG_WORK=work_ex
   python pose_body.py                 # work_ex/pose_body.json, pose_L/ pose_R/ mask_L/ mask_R/ matte_isnet/
   python face_crops.py --side L --from-pose   # work_ex/face_L/ (face box from pose_body.json; or run track.py first)
   ```
3. Reference plate of the replacement character:
   ```
   python make_refs.py cat --photo <photo of the cat sitting upright> --out refs/cat_body_base.png
   ```
   `refs/cat_body_base_hq.png` is what `run_example.ps1` loads.
4. Render, always probe first:
   ```
   powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side cat -Probe
   powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side cat
   ```
   Frames land in `work_ex/wan/cat_E/frames/` (12 fps, 768x576), plus `cat_E.mp4`.
5. Post:
   ```
   python deflicker.py --in work_ex/wan/cat_E/frames --src work_ex/wan/cat_E/in/bg --out work_ex/wan/cat_E/frames_df   # only for renders made with --decode-frames 16
   python to30.py --in work_ex/wan/cat_E/frames --out work_ex/cat24 --method rife --gpu 0 --n-out 361
   python mouth_curve.py --vocals <isolated vocals wav> --fps 24 --n 361 --out work_ex/mouth
   .venv-lp/Scripts/python.exe lip_pass.py calibrate --frames work_ex/cat24           # once per cat: needs faces/cat open-mouth photos
   .venv-lp/Scripts/python.exe lip_pass.py run --frames work_ex/cat24 --out work_ex/cat24_lips --strength 1.2
   ffmpeg -framerate 24 -i work_ex/cat24_lips/%05d.png -vf scale=960:720:flags=lanczos -c:v libx264 -crf 16 -pix_fmt yuv420p work_ex/wan/cat24_lips.mp4
   python finish.py --blocks work_ex/blocks.json --input work_ex/wan/cat24_lips.mp4 --out work_ex/final.mp4
   ```
   `finish.py` re-muxes the original audio, composes the 9:16 layout and checks audio/video sync.

## B. Your own clip

The scripts index everything by *cut frame number* and work on one **block** (a contiguous
frame range) and one **side** (L or R, by bounding-box centre) at a time.

1. `work/cut.mp4` + `work/audio.m4a` as above. Keep clips short (the demo is 361 frames) and
   performers on fixed sides of the frame.
2. Write `work/blocks.json` by hand:
   ```json
   {
     "fps": 24,
     "blocks": [{"name": "E", "src": "work/cut.mp4", "cut_start": 0, "cut_end": 361, "n": 361, "out_start": 0}],
     "n_total": 361,
     "audio": "work/audio.m4a", "ref": "work/cut.mp4", "song": "work/audio.m4a", "cut": "work/cut.mp4",
     "song_offset": 0.0,
     "sheet_times": [1, 4, 8, 12]
   }
   ```
   `song_offset` is the clip's video-minus-audio start offset (`ffprobe -show_entries stream=start_time`).
3. `pose_body.py --cuts <comma-separated shot-cut frame numbers>` (the default list is the demo clip's), `face_crops.py --side <L|R> --from-pose`, and a reference plate with `make_refs.py user`
   (human, from full-body photos with a large face) or `make_refs.py cat` (animal).
   For an animal replacing a human, also run `cat_pose.py --side <side>` and pass
   `--pose-dir work/pose_<side>_cat` so the skeleton has cat proportions.
4. Render directly with the driver instead of the demo wrapper:
   ```
   python wan_animate.py --block E --side L --ref refs/your_ref.png --name take1 ^
       --prompt "<who the character is, what they wear, where they sit, fixed camera>" ^
       --negative-extra "<what must not appear>" ^
       --width 768 --height 576 --length 33 --stride 2 --probe 33
   ```
   Drop `--probe 33` for the full render; add `--resume` to continue after an interruption.
   Width and height must be multiples of 16 and match the clip's aspect; `--length` must be 4k+1.
   `--mask-dir` and `--block-px` control how much of the source is repainted (see
   `make_fur_mask.py` for costume-preserving masks).
5. Post passes as in A.5, with `--n-out` = 2 x rendered frames - 1 (or the clip's frame count).

## Tips that saved hours

- Probe every new reference or prompt (`--probe 33`, ~15 min on an iGPU) before a full render.
- `-ReserveVram 4` / `--reserve-vram 4`: with 16 GB the 11 GB model must load partially or the
  sampler OOMs two minutes in. The ComfyUI log must say `loaded partially`.
- Never reuse a ComfyUI server left by another job; stale VRAM leads to OOM.
- "device lost" on an iGPU: re-run the same command, finished windows are skipped.
- A human reference needs a big face in the plate (Studio > References shows the pixel size);
  a 40 px face drifts to a different person at every shot cut.
