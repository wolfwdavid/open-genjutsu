<script>
  import { onMount } from 'svelte';
  import { ImagePlus, Trash2, Wand2, CheckCircle2, AlertTriangle, ArrowRightLeft } from 'lucide-svelte';
  import { api, fileUrl, fmtBytes, fmtClock } from '../lib/api.js';

  let refs = $state(null);
  let error = $state(null);
  let uploading = $state(false);
  let dragOver = $state(false);
  let job = $state(null);
  let promoting = $state(false);
  let confirmPromote = $state(false);
  let fileInput = $state(null);

  async function load() { try { refs = await api.refs(); error = null; } catch (e) { error = e.message; } }
  onMount(load);

  async function addFiles(list) {
    const files = [...list].filter(f => f.type.startsWith('image/') || /\.(heic|heif)$/i.test(f.name));
    if (!files.length) return;
    uploading = true; error = null;
    try { await api.uploadRefs(files); await load(); } catch (e) { error = e.message; } finally { uploading = false; }
  }
  async function remove(u) {
    try { await api.deleteUpload(u.path.split('/').pop()); await load(); } catch (e) { error = e.message; }
  }
  async function build() {
    error = null; confirmPromote = false;
    try {
      const { job: id } = await api.buildRef();
      job = { id, state: 'running', lines: [] };
      const poll = async () => {
        job = await api.job(id);
        if (job.state === 'running') setTimeout(poll, 1500); else await load();
      };
      poll();
    } catch (e) { error = e.message; }
  }
  async function promote() {
    promoting = true; error = null;
    try { await api.promoteRef(); confirmPromote = false; job = null; await load(); } catch (e) { error = e.message; } finally { promoting = false; }
  }
  const best = $derived(job?.result?.best);
  const faceVerdict = $derived.by(() => {
    if (!best) return null;
    if (best.face_h >= 300) return { cls: 'text-ok', text: `Face ${best.face_h} px tall in the photo: good identity detail.` };
    if (best.face_h >= 150) return { cls: 'text-warn', text: `Face ${best.face_h} px tall: usable, a closer photo would be better.` };
    return { cls: 'text-danger', text: `Face only ${best.face_h} px tall: identity will drift again. Use a closer photo.` };
  });
</script>

<div class="max-w-6xl mx-auto grid gap-4">
  <header>
    <h1 class="text-xl font-semibold">Reference photos</h1>
    <p class="text-sm text-fg-muted">Build <code class="font-mono">refs/user_ref.png</code> from close-up photos. The render takes your identity only from this image, so the face must be large and sharp. No sunglasses needed: the prompt adds them.</p>
  </header>

  {#if error}<div class="panel p-3 border-danger/50 text-danger flex gap-2"><AlertTriangle size={18} /> {error}</div>{/if}

  <div class="grid lg:grid-cols-[1fr_1.4fr] gap-4">
    <!-- current reference -->
    <section class="panel p-4 grid gap-3 content-start">
      <div class="label">Current reference</div>
      {#if refs?.current}
        <img src={fileUrl(refs.current.path) + '&v=' + refs.current.mtime} alt="Current user reference" class="rounded-md border border-line max-h-[420px] w-auto object-contain justify-self-start bg-bg" />
        <div class="text-sm text-fg-muted font-mono tnum">{refs.current.width}×{refs.current.height} · {fmtBytes(refs.current.bytes)} · {new Date(refs.current.mtime * 1000).toLocaleDateString()}</div>
        {#if refs.current.width < 400}
          <div class="text-sm text-warn flex gap-2"><AlertTriangle size={16} class="shrink-0 mt-0.5" /> A full-body plate this narrow leaves the face around 40 px: that is what made the identity drift between shots.</div>
        {/if}
      {:else}<div class="text-fg-muted">none yet</div>{/if}
      {#if refs?.previous?.length}
        <div class="label mt-2">Previous</div>
        <div class="flex gap-2 flex-wrap">{#each refs.previous as p}<img src={fileUrl(p.path)} alt="previous reference" class="h-20 rounded border border-line" title={p.path} />{/each}</div>
      {/if}
    </section>

    <!-- photos + build -->
    <section class="grid gap-4 content-start">
      <div class="panel p-4 grid gap-3">
        <div class="flex items-center justify-between"><div class="label">Photos ({refs?.uploads?.length ?? 0})</div>
          <button class="btn" onclick={() => fileInput.click()} disabled={uploading}><ImagePlus size={16} /> Add photos</button>
          <input bind:this={fileInput} type="file" accept="image/*,.heic,.heif" multiple class="hidden" onchange={(e) => { addFiles(e.target.files); e.target.value = ''; }} />
        </div>
        <div role="button" tabindex="0" aria-label="Drop photos here"
             class="rounded-md border-2 border-dashed p-5 text-center text-fg-muted transition-colors {dragOver ? 'border-accent bg-accent/10 text-fg' : 'border-line-strong'}"
             ondragover={(e) => { e.preventDefault(); dragOver = true; }} ondragleave={() => dragOver = false}
             ondrop={(e) => { e.preventDefault(); dragOver = false; addFiles(e.dataTransfer.files); }}
             onclick={() => fileInput.click()} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); }}>
          {uploading ? 'Uploading…' : 'Drop 2–3 close-ups here (front, ¾, side; face filling the frame; even light; the black suit if possible). JPG, PNG or HEIC.'}
        </div>
        {#if refs?.uploads?.length}
          <ul class="grid grid-cols-3 sm:grid-cols-4 gap-2">
            {#each refs.uploads as u (u.path)}
              <li class="relative group">
                <img src={fileUrl(u.path)} alt={u.path.split('/').pop()} class="aspect-[3/4] w-full object-cover rounded-md border border-line bg-bg" loading="lazy" />
                <div class="absolute bottom-1 left-1 text-[11px] font-mono bg-bg/80 px-1 rounded">{u.width}×{u.height}</div>
                <button class="absolute top-1 right-1 size-8 grid place-items-center rounded-md bg-bg/80 text-danger opacity-80 hover:opacity-100" aria-label="Remove photo" onclick={() => remove(u)}><Trash2 size={15} /></button>
              </li>
            {/each}
          </ul>
        {/if}
        <div class="flex items-center gap-3">
          <button class="btn btn-primary" onclick={build} disabled={!refs?.uploads?.length || job?.state === 'running'}><Wand2 size={16} /> {job?.state === 'running' ? 'Building…' : 'Build reference'}</button>
          <span class="text-sm text-fg-dim">runs make_refs.py user (face detect + person matte), ~1 min</span>
        </div>
      </div>

      {#if job}
        <div class="panel p-4 grid gap-3">
          <div class="flex items-center gap-2">
            {#if job.state === 'running'}<span class="size-2 rounded-full bg-accent animate-pulse"></span><span>Building…</span>
            {:else if job.state === 'done'}<CheckCircle2 class="text-ok" size={18} /><span>Candidate ready</span>
            {:else}<AlertTriangle class="text-danger" size={18} /><span>Build failed (exit {job.rc})</span>{/if}
          </div>
          {#if job.state === 'done' && job.result?.candidate}
            <div class="grid sm:grid-cols-2 gap-3">
              <div><div class="label mb-1">Current</div>{#if refs?.current}<img src={fileUrl(refs.current.path) + '&v=' + refs.current.mtime} alt="current" class="rounded-md border border-line max-h-72 bg-bg" />{/if}</div>
              <div><div class="label mb-1">Candidate <ArrowRightLeft size={12} class="inline" /></div><img src={fileUrl(job.result.candidate.path) + '&v=' + job.result.candidate.mtime} alt="candidate" class="rounded-md border border-accent max-h-72 bg-bg" /></div>
            </div>
            {#if faceVerdict}<div class="text-sm {faceVerdict.cls}">{faceVerdict.text} Best photo: <span class="font-mono">{best.photo}</span> (score {best.score}, yaw {best.yaw}°).</div>{/if}
            {#if job.result.sheet}<a class="text-sm text-info underline" href={fileUrl(job.result.sheet.path)} target="_blank" rel="noopener">Open the ranking sheet</a>{/if}
            <div class="flex flex-wrap items-center gap-3">
              {#if !confirmPromote}
                <button class="btn btn-primary" onclick={() => confirmPromote = true}>Use as reference</button>
              {:else}
                <span class="text-sm">Replace refs/user_ref.png? The old one is kept as user_ref_prev_*.png.</span>
                <button class="btn btn-primary" onclick={promote} disabled={promoting}>{promoting ? 'Replacing…' : 'Yes, replace'}</button>
                <button class="btn" onclick={() => confirmPromote = false}>Cancel</button>
              {/if}
            </div>
          {/if}
          <details class="text-sm" open={job.state === 'failed'}>
            <summary class="text-fg-muted cursor-pointer">make_refs output</summary>
            <pre class="font-mono text-[12px] mt-2 max-h-56 overflow-auto text-fg-muted whitespace-pre-wrap">{job.lines.join('\n')}</pre>
          </details>
        </div>
      {/if}
    </section>
  </div>
</div>
