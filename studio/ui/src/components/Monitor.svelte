<script>
  import { AlertTriangle, CheckCircle2, Cpu, HardDrive, MemoryStick, RefreshCw, TerminalSquare } from 'lucide-svelte';
  import { live, refresh, selectLog } from '../lib/status.svelte.js';
  import { fmtClock } from '../lib/api.js';

  const run = $derived(live.data?.run);
  const comfy = $derived(live.data?.comfy);
  const procs = $derived(live.data?.procs);
  const logs = $derived(live.data?.logs ?? []);
  let autoScroll = $state(true);
  let logEl = $state(null);

  $effect(() => {
    if (autoScroll && logEl && run?.tail) logEl.scrollTop = logEl.scrollHeight;
  });

  const stateText = { running: 'Rendering', done: 'Finished', failed: 'Failed', stalled: 'Stalled', idle: 'Idle' };
  const stateCls = { running: 'text-accent-strong', done: 'text-ok', failed: 'text-danger', stalled: 'text-warn', idle: 'text-fg-muted' };

  function windowStyle(w) {
    if (w.state === 'done' || w.state === 'skipped') return 'bg-accent';
    if (w.state === 'running') return 'bg-accent/40 striped';
    return 'bg-line-strong';
  }
  function windowTitle(w) {
    const base = `window ${w.index}: frames ${w.seq}–${w.seq + w.length - 1} (${w.new} new)`;
    if (w.state === 'done') return `${base} · done in ${w.seconds}s, ${w.spf} s/frame`;
    if (w.state === 'skipped') return `${base} · already rendered`;
    if (w.state === 'running') return `${base} · in progress`;
    return `${base} · pending`;
  }
  const stepPct = $derived(comfy?.step && run?.state === 'running' ? Math.round(100 * comfy.step.k / comfy.step.n) : null);
</script>

<div class="max-w-6xl mx-auto grid gap-4">
  <header class="flex flex-wrap items-end justify-between gap-3">
    <div>
      <h1 class="text-xl font-semibold">Render monitor</h1>
      <p class="text-sm text-fg-muted">Reads work_ex/round4/render_*.log and the ComfyUI server log. Updates every 3 s.</p>
    </div>
    <div class="flex items-center gap-2">
      {#if logs.length > 1}
        <label class="label" for="logsel">Run</label>
        <select id="logsel" class="input !w-auto" value={live.selectedLog ?? logs[0]?.file} onchange={(e) => selectLog(e.target.value)}>
          {#each logs as l}<option value={l.file}>{l.label} · {fmtClock(l.mtime)}</option>{/each}
        </select>
      {/if}
      <button class="btn" onclick={refresh} aria-label="Refresh now"><RefreshCw size={16} /> Refresh</button>
    </div>
  </header>

  {#if live.error}
    <div class="panel p-4 border-danger/50 text-danger flex gap-3"><AlertTriangle /> Backend unreachable: {live.error}</div>
  {:else if !live.data}
    <div class="panel p-6 animate-pulse text-fg-dim">Loading status…</div>
  {:else if !run}
    <div class="panel p-6 grid gap-3">
      <div class="text-lg font-medium">No render has been started yet</div>
      <p class="text-fg-muted">Start one from your own PowerShell (renders must not run under an agent shell):</p>
      <pre class="card p-3 font-mono text-sm overflow-auto">powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side user -Probe   # 15 min test
powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side user          # full pass, ~1 h 40 min</pre>
      <p class="text-fg-muted text-sm">This page picks the run up automatically once the log appears.</p>
    </div>
  {:else}
    <!-- hero -->
    <section class="panel p-5 grid gap-5" aria-live="polite">
      <div class="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div class="label">{run.label}</div>
          <div class="text-3xl font-semibold {stateCls[run.state]}">{stateText[run.state] ?? run.state}</div>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-2 text-right">
          <div><div class="label">Frames</div><div class="font-mono text-lg tnum">{run.done_frames}<span class="text-fg-dim">/{run.planned_frames}</span></div></div>
          <div><div class="label">s / frame</div><div class="font-mono text-lg tnum">{run.spf ? run.spf.toFixed(1) : '—'}</div></div>
          <div><div class="label">Elapsed</div><div class="font-mono text-lg tnum">{run.elapsed_text ?? '—'}</div></div>
          <div><div class="label">ETA</div><div class="font-mono text-lg tnum">{run.eta_text ?? '—'}{#if run.eta_clock}<span class="text-fg-dim text-sm"> · {run.eta_clock}</span>{/if}</div></div>
        </div>
      </div>

      <!-- windows bar -->
      <div>
        <div class="flex justify-between text-xs text-fg-dim mb-1"><span>Windows ({run.windows.length})</span><span class="font-mono tnum">{Math.round((run.progress ?? 0) * 100)}%</span></div>
        <div class="flex gap-[3px] h-4 rounded-md overflow-hidden" role="progressbar" aria-valuenow={Math.round((run.progress ?? 0) * 100)} aria-valuemin="0" aria-valuemax="100">
          {#each run.windows as w}
            <div class="h-full rounded-[2px] {windowStyle(w)}" style="flex: {Math.max(1, w.new)} 1 0" title={windowTitle(w)}></div>
          {/each}
        </div>
        {#if run.current}
          <div class="mt-2 text-sm text-fg-muted flex flex-wrap gap-x-4 gap-y-1">
            <span>Window <b class="text-fg font-mono">{run.current.index}</b> · frames {run.current.seq}–{run.current.seq + run.current.length - 1} ({run.current.new} new)</span>
            {#if comfy?.step}
              <span>Sampler step <b class="text-fg font-mono">{comfy.step.k}/{comfy.step.n}</b> · {comfy.step.s_per_it.toFixed(0)} s/step · {comfy.step.eta} left in this window</span>
            {/if}
            {#if run.current.started}<span>started {fmtClock(run.current.started)}</span>{/if}
          </div>
          {#if stepPct !== null}
            <div class="mt-1 h-1.5 rounded bg-line-strong overflow-hidden"><div class="h-full bg-info transition-[width] duration-500" style="width:{stepPct}%"></div></div>
          {/if}
        {/if}
      </div>

      <!-- alerts -->
      {#if run.alerts.length}
        <ul class="grid gap-2">
          {#each run.alerts as a}
            <li class="card p-3 flex gap-3 {a.level === 'error' ? 'border-danger/60' : 'border-warn/60'}">
              <AlertTriangle class={a.level === 'error' ? 'text-danger shrink-0' : 'text-warn shrink-0'} size={20} />
              <div><div class="font-medium">{a.text}</div><div class="text-sm text-fg-muted">{a.action}</div></div>
            </li>
          {/each}
        </ul>
      {:else if run.state === 'done'}
        <div class="card p-3 flex gap-3 border-ok/50"><CheckCircle2 class="text-ok shrink-0" size={20} />
          <div><div class="font-medium">{run.total_line ?? 'Render finished.'}</div><div class="text-sm text-fg-muted">{run.frames_on_disk} frames in {run.frames_dir}. Next: deflicker → RIFE → lip pass → finish (see RUNBOOK).</div></div></div>
      {/if}
    </section>

    <!-- system row -->
    <section class="grid sm:grid-cols-3 gap-3">
      <div class="card p-4 flex items-start gap-3">
        <Cpu class="shrink-0 text-fg-dim" size={20} />
        <div>
          <div class="label">Wan model load</div>
          {#if comfy?.load}
            <div class="font-medium {comfy.load.mode === 'partially' ? 'text-ok' : comfy.oom_count ? 'text-warn' : ''}">{comfy.load.mode === 'partially' ? 'Partial (safe)' : comfy.oom_count ? 'Complete (caused OOM)' : 'Complete'}</div>
            <div class="text-sm text-fg-muted font-mono tnum">{(comfy.load.loaded_mb / 1024).toFixed(1)} GB on GPU · {(comfy.load.offloaded_mb / 1024).toFixed(1)} GB offloaded</div>
          {:else}<div class="text-fg-muted">not loaded yet</div>{/if}
        </div>
      </div>
      <div class="card p-4 flex items-start gap-3">
        <HardDrive class="shrink-0 text-fg-dim" size={20} />
        <div>
          <div class="label">Processes</div>
          <div class="flex gap-2 mt-1">
            <span class="pill {procs?.comfy ? 'pill-ok' : ''}">ComfyUI {procs?.comfy ? 'up' : 'down'}</span>
            <span class="pill {procs?.driver ? 'pill-ok' : ''}">render driver {procs?.driver ? 'up' : 'down'}</span>
          </div>
          {#if comfy?.oom_count}<div class="text-sm text-danger mt-1">{comfy.oom_count} OOM event{comfy.oom_count > 1 ? 's' : ''} this server session</div>{/if}
        </div>
      </div>
      <div class="card p-4 flex items-start gap-3">
        <MemoryStick class="shrink-0 text-fg-dim" size={20} />
        <div>
          <div class="label">System RAM</div>
          <div class="font-medium font-mono tnum {procs && procs.ram_free_gb < 2 ? 'text-warn' : ''}">{procs?.ram_free_gb} GB free <span class="text-fg-dim">/ {procs?.ram_total_gb} GB</span></div>
          <div class="text-sm text-fg-muted">below 2 GB the render can be killed</div>
        </div>
      </div>
    </section>

    <!-- log tail -->
    <section class="panel">
      <div class="flex items-center justify-between px-4 py-2 border-b border-line">
        <div class="flex items-center gap-2 text-sm text-fg-muted"><TerminalSquare size={16} /> {run.log}</div>
        <label class="text-sm text-fg-muted flex items-center gap-2"><input type="checkbox" bind:checked={autoScroll} /> follow</label>
      </div>
      <pre bind:this={logEl} class="font-mono text-[12.5px] leading-relaxed p-4 max-h-80 overflow-auto text-fg-muted whitespace-pre-wrap">{run.tail.join('\n')}</pre>
    </section>
  {/if}
</div>
