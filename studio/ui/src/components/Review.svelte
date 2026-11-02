<script>
  import { onMount } from 'svelte';
  import { ChevronLeft, ChevronRight, ThumbsUp, RotateCcw, Send, X, AlertTriangle, ExternalLink, Copy } from 'lucide-svelte';
  import { api, frameUrl, fileUrl, tc, fmtBytes } from '../lib/api.js';

  let data = $state(null);
  let error = $state(null);
  let shotN = $state(1);
  let laneA = $state('source');
  let laneB = $state('cat_df');
  let mode = $state('side');       // side | wipe
  let wipe = $state(50);
  let frame = $state(0);           // strided index (absolute)
  let note = $state('');
  let saving = $state(false);
  let videos = $state([]);
  let player = $state(null);
  let playerSrc = $state(null);
  let handoff = $state(null);      // { open, video, name, busy, result, error }

  const shot = $derived(data?.shots.find(s => s.n === shotN));
  const lanes = $derived(data?.lanes ?? {});
  const laneKeys = $derived(Object.keys(lanes));
  const available = $derived(shot ? shot.lanes : []);

  async function load() {
    try {
      data = await api.shots();
      videos = (await api.videos()).videos;
      if (!playerSrc) playerSrc = videos.find(v => v.name === 'final_cat_lips.mp4')?.path ?? videos[0]?.path ?? null;
      pickDefaults();
    } catch (e) { error = e.message; }
  }
  function pickDefaults() {
    if (!shot) return;
    if (!available.includes(laneA)) laneA = available[0] ?? '';
    if (!available.includes(laneB)) laneB = available.find(k => k !== laneA) ?? laneA;
    if (frame < shot.start || frame >= shot.end) frame = shot.start;
    note = shot.review?.note ?? '';
  }
  onMount(load);
  function selectShot(n) { shotN = n; frame = data.shots[n - 1].start; pickDefaults(); seekPlayer(); }
  function step(d) { if (!shot) return; frame = Math.min(shot.end - 1, Math.max(shot.start, frame + d)); }
  function seekPlayer() { if (player && shot) { player.currentTime = shot.t_start + 0.02; } }

  async function decide(decision) {
    if (!shot) return;
    saving = true;
    try {
      const next = shot.review?.decision === decision ? null : decision;
      await api.review(shot.n, { decision: next, note });
      await load();
    } catch (e) { error = e.message; } finally { saving = false; }
  }
  async function saveNote() {
    if (!shot) return;
    saving = true;
    try { await api.review(shot.n, { decision: shot.review?.decision ?? null, note }); await load(); } catch (e) { error = e.message; } finally { saving = false; }
  }

  function openHandoff() {
    handoff = { open: true, video: playerSrc ?? videos[0]?.path, name: '', busy: false, result: null, error: null };
  }
  async function sendToEditor() {
    handoff.busy = true; handoff.error = null;
    try { handoff.result = await api.handoff({ video: handoff.video, name: handoff.name || null }); }
    catch (e) { handoff.error = e.message; } finally { handoff.busy = false; }
  }
  function onKey(e) {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || handoff?.open) return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); step(e.shiftKey ? -5 : -1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(e.shiftKey ? 5 : 1); }
    else if (e.key === 'a') decide('approve');
    else if (e.key === 'r') decide('reroll');
    else if (e.key === 'w') mode = mode === 'wipe' ? 'side' : 'wipe';
  }
  const decisionPill = (r) => r?.decision === 'approve' ? 'pill-ok' : r?.decision === 'reroll' ? 'pill-warn' : '';
  const decisionText = (r) => r?.decision === 'approve' ? 'approved' : r?.decision === 'reroll' ? 're-roll' : 'unreviewed';
</script>

<svelte:window onkeydown={onKey} />

<div class="max-w-7xl mx-auto grid gap-4">
  <header class="flex flex-wrap items-end justify-between gap-3">
    <div>
      <h1 class="text-xl font-semibold">Shot review</h1>
      <p class="text-sm text-fg-muted">Compare any two frame lanes per camera shot. Keys: ← → frame (Shift: ×5), A approve, R re-roll, W wipe.</p>
    </div>
    <button class="btn btn-primary" onclick={openHandoff} disabled={!videos.length}><Send size={16} /> Send to editor</button>
  </header>

  {#if error}<div class="panel p-3 border-danger/50 text-danger flex gap-2"><AlertTriangle size={18} /> {error}</div>{/if}

  {#if data}
    <!-- shot strip -->
    <section class="panel p-3" aria-label="Shots">
      <div class="flex gap-1 h-14" role="tablist">
        {#each data.shots as s}
          <button role="tab" aria-selected={s.n === shotN} onclick={() => selectShot(s.n)}
                  class="relative rounded-md border px-2 text-left overflow-hidden transition-colors {s.n === shotN ? 'border-accent bg-accent/10' : 'border-line hover:border-line-strong bg-raised'}"
                  style="flex: {s.frames} 1 0">
            <div class="text-sm font-medium">Shot {s.n}</div>
            <div class="text-[11px] font-mono tnum text-fg-dim">{tc(s.cut_start)}–{tc(s.cut_end)}</div>
            {#if s.review?.decision}<span class="absolute top-1.5 right-1.5 size-2 rounded-full {s.review.decision === 'approve' ? 'bg-ok' : 'bg-warn'}"></span>{/if}
          </button>
        {/each}
      </div>
    </section>

    <div class="grid xl:grid-cols-[1fr_300px] gap-4">
      <!-- compare -->
      <section class="panel p-4 grid gap-3">
        <div class="flex flex-wrap items-center gap-3">
          <label class="flex items-center gap-2 text-sm"><span class="label">A</span>
            <select class="input !w-auto" bind:value={laneA}>{#each laneKeys as k}<option value={k} disabled={!available.includes(k)}>{lanes[k].label}</option>{/each}</select></label>
          <label class="flex items-center gap-2 text-sm"><span class="label">B</span>
            <select class="input !w-auto" bind:value={laneB}>{#each laneKeys as k}<option value={k} disabled={!available.includes(k)}>{lanes[k].label}</option>{/each}</select></label>
          <div class="ml-auto flex rounded-md border border-line-strong overflow-hidden" role="group" aria-label="Compare mode">
            <button class="px-3 h-9 text-sm {mode === 'side' ? 'bg-raised text-fg' : 'text-fg-muted'}" onclick={() => mode = 'side'}>Side by side</button>
            <button class="px-3 h-9 text-sm border-l border-line-strong {mode === 'wipe' ? 'bg-raised text-fg' : 'text-fg-muted'}" onclick={() => mode = 'wipe'}>Wipe</button>
          </div>
        </div>

        {#if shot && laneA && laneB}
          {#if mode === 'side'}
            <div class="grid md:grid-cols-2 gap-2">
              {#each [laneA, laneB] as k, i}
                <figure class="relative">
                  <img src={frameUrl(k, frame, 768)} alt="{lanes[k].label}, frame {frame}" class="w-full rounded-md border border-line bg-bg aspect-[4/3] object-contain" />
                  <figcaption class="absolute top-2 left-2 pill bg-bg/80">{i ? 'B' : 'A'} · {lanes[k].label}</figcaption>
                </figure>
              {/each}
            </div>
          {:else}
            <div class="relative select-none">
              <img src={frameUrl(laneB, frame, 1024)} alt="{lanes[laneB].label}, frame {frame}" class="w-full rounded-md border border-line bg-bg aspect-[4/3] object-contain" />
              <img src={frameUrl(laneA, frame, 1024)} alt="{lanes[laneA].label}, frame {frame}" class="absolute inset-0 w-full h-full rounded-md object-contain bg-bg" style="clip-path: inset(0 {100 - wipe}% 0 0)" />
              <div class="absolute inset-y-0 w-0.5 bg-accent" style="left:{wipe}%"></div>
              <span class="absolute top-2 left-2 pill bg-bg/80">A · {lanes[laneA].label}</span>
              <span class="absolute top-2 right-2 pill bg-bg/80">B · {lanes[laneB].label}</span>
              <input type="range" min="0" max="100" bind:value={wipe} aria-label="Wipe position" class="absolute inset-x-0 bottom-2 w-full accent-[var(--color-accent)]" />
            </div>
          {/if}

          <!-- scrubber -->
          <div class="flex items-center gap-3">
            <button class="btn !min-h-9 !px-2" onclick={() => step(-1)} aria-label="Previous frame"><ChevronLeft size={18} /></button>
            <input type="range" min={shot.start} max={shot.end - 1} bind:value={frame} class="flex-1 accent-[var(--color-accent)]" aria-label="Frame within shot" />
            <button class="btn !min-h-9 !px-2" onclick={() => step(1)} aria-label="Next frame"><ChevronRight size={18} /></button>
            <div class="font-mono tnum text-sm w-40 text-right"><span class="text-fg">{tc(frame * data.stride)}</span> <span class="text-fg-dim">· f{frame}/{data.total - 1}</span></div>
          </div>
        {/if}

        <!-- player -->
        <div class="grid gap-2">
          <div class="flex items-center gap-3">
            <span class="label">Preview</span>
            <select class="input !w-auto" bind:value={playerSrc}>{#each videos as v}<option value={v.path}>{v.name} · {fmtBytes(v.bytes)}</option>{/each}</select>
            <button class="btn !min-h-9" onclick={seekPlayer} disabled={!shot}>Jump to shot {shotN}</button>
          </div>
          {#if playerSrc}
            <!-- svelte-ignore a11y_media_has_caption -->
            <video bind:this={player} src={fileUrl(playerSrc)} controls preload="metadata" class="w-full max-h-[420px] rounded-md border border-line bg-black"></video>
          {/if}
        </div>
      </section>

      <!-- decision rail -->
      <aside class="panel p-4 grid gap-4 content-start">
        {#if shot}
          <div>
            <div class="label">Shot {shot.n}</div>
            <div class="font-mono tnum text-sm text-fg-muted">{tc(shot.cut_start)} – {tc(shot.cut_end)} · {shot.frames} rendered frames</div>
            <span class="pill mt-2 {decisionPill(shot.review)}">{decisionText(shot.review)}</span>
          </div>
          <div class="grid grid-cols-2 gap-2">
            <button class="btn btn-ok {shot.review?.decision === 'approve' ? '!bg-ok/20' : ''}" onclick={() => decide('approve')} disabled={saving}><ThumbsUp size={16} /> Approve</button>
            <button class="btn btn-danger {shot.review?.decision === 'reroll' ? '!bg-danger/20' : ''}" onclick={() => decide('reroll')} disabled={saving}><RotateCcw size={16} /> Re-roll</button>
          </div>
          <label class="grid gap-1 text-sm">
            <span class="label">Note</span>
            <textarea class="input !min-h-24 py-2" bind:value={note} placeholder="e.g. pink paw smear at 6.5 s, left of the stool" onblur={saveNote}></textarea>
          </label>
          <div>
            <div class="label mb-1">Lanes with this shot</div>
            <ul class="text-sm text-fg-muted grid gap-0.5">{#each available as k}<li>{lanes[k].label} <span class="text-fg-dim font-mono">{lanes[k].dir}</span></li>{/each}</ul>
          </div>
          {#if shot.review?.decision === 'reroll'}
            <div class="card p-3 text-sm">
              <div class="label mb-1">Re-render command</div>
              <code class="font-mono text-[12px] break-all">powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side cat -FixShot {shot.n}</code>
              <div class="text-fg-dim mt-1">~23 min; splices itself back and refreshes the deflickered lane.</div>
            </div>
          {/if}
        {/if}
      </aside>
    </div>
  {:else if !error}
    <div class="panel p-6 animate-pulse text-fg-dim">Loading shots…</div>
  {/if}
</div>

{#if handoff?.open}
  <div class="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-labelledby="handoff-title">
    <div class="panel w-full max-w-lg p-5 grid gap-4">
      <div class="flex items-center justify-between">
        <h2 id="handoff-title" class="text-lg font-semibold">Send to editor</h2>
        <button class="btn !min-h-9 !px-2" aria-label="Close" onclick={() => handoff = null}><X size={18} /></button>
      </div>
      {#if !handoff.result}
        <label class="grid gap-1 text-sm"><span class="label">Video</span>
          <select class="input" bind:value={handoff.video}>{#each videos as v}<option value={v.path}>{v.name} · {fmtBytes(v.bytes)}</option>{/each}</select></label>
        <label class="grid gap-1 text-sm"><span class="label">Project name (optional)</span>
          <input class="input" bind:value={handoff.name} placeholder="Cat duet 2026-10-04 15:00" /></label>
        <p class="text-sm text-fg-muted">Creates a project in video_editor_ai and uploads the file. Without a saved editor session (Settings) it stages the file and opens the editor instead.</p>
        {#if handoff.error}<div class="text-danger text-sm">{handoff.error}</div>{/if}
        <div class="flex justify-end gap-2">
          <button class="btn" onclick={() => handoff = null}>Cancel</button>
          <button class="btn btn-primary" onclick={sendToEditor} disabled={handoff.busy || !handoff.video}><Send size={16} /> {handoff.busy ? 'Sending…' : 'Send'}</button>
        </div>
      {:else}
        {#if handoff.result.mode === 'api'}
          <div class="text-ok">Project created and the video uploaded.</div>
          <a class="btn btn-primary justify-self-start" href={handoff.result.project_url} target="_blank" rel="noopener"><ExternalLink size={16} /> Open in editor</a>
        {:else}
          <div class="text-warn flex gap-2"><AlertTriangle size={18} class="shrink-0 mt-0.5" /> {handoff.result.reason}</div>
          <div class="text-sm">The file is staged at</div>
          <div class="card p-2 font-mono text-[12px] break-all flex items-center gap-2">{handoff.result.staged}
            <button class="btn !min-h-8 !px-2 ml-auto" aria-label="Copy path" onclick={() => navigator.clipboard?.writeText(handoff.result.staged)}><Copy size={14} /></button></div>
          <a class="btn justify-self-start" href={handoff.result.projects_url} target="_blank" rel="noopener"><ExternalLink size={16} /> Open the editor</a>
        {/if}
        <div class="flex justify-end"><button class="btn" onclick={() => handoff = null}>Done</button></div>
      {/if}
    </div>
  </div>
{/if}
