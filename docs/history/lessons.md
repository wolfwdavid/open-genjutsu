# Lessons

## 2026-09-25: match the reference, not the cheapest lane
- Correction: after three rounds on a head-swap composite, the user said "I want a rebuild not just a face swap (body swap)" and pointed back at the Genjutsu reference.
- Pattern: the user's reference video defines the deliverable. A lane that can only reach a fraction of that look (head-on-body composite) should be flagged as such before investing QA rounds in it.
- Rule: before building, write one line comparing the planned output to the reference ("example: whole cat on a stool in costume; this lane: cat head on the rapper's body"). If they differ in kind, ask which one the user wants.

## 2026-09-26: generative recast of an animal from a human driver
- Wan2.2-Animate replacement with a human skeleton renders a human, even with a cat reference: the costume reference gave a human in cap/flannel, the bare cat gave a cat head on a human torso, and dropping face conditioning gave a fully human figure.
- Face conditioning is what keeps the cat face; keep it on. Retarget the skeleton to animal proportions (no fingers, short forelegs, short seated torso) before judging the model.
- Probe every character with one 33-frame window (13 min) before any full render.
- RESULT (2026-09-26, probe_R_A_catpose): bare cat reference + face conditioning + cat_pose.py skeleton (+ anti-human negative) = a real cat on a wooden stool, black sunglasses from the prompt, paws following the rapper's hands. The skeleton retarget was the fix; the prompt alone did not add the cap/flannel.

## 2026-09-27: review a finished render as a whole before post-processing it
- The user pass finished and the plan went straight to RIFE/lip/finish. RIFE flagging 17 "cuts" was the only reason
  the per-second decode flicker (present since the first cat render) and the per-shot identity drift got caught.
- Rule: after every render, before any downstream step, check (a) brightness per frame vs the source (periodic pulses =
  decode chunking), (b) one crop per shot of each character next to its reference (identity), (c) outfit vs prompt.
- Rule: when a memory fix fails once, read the loader line ("loaded completely/partially") before retrying a guess;
  the 2 GB reserve cost one extra 10-min attempt that the log would have ruled out.
