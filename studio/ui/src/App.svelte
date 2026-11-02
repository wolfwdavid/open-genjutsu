<script>
  import { onMount } from 'svelte';
  import { Activity, Images, Film, Settings as SettingsIcon, Clapperboard } from 'lucide-svelte';
  import { live, startPolling } from './lib/status.svelte.js';
  import Monitor from './components/Monitor.svelte';
  import References from './components/References.svelte';
  import Review from './components/Review.svelte';
  import Settings from './components/Settings.svelte';

  const views = [
    { id: 'monitor', label: 'Monitor', icon: Activity, key: '1' },
    { id: 'refs', label: 'References', icon: Images, key: '2' },
    { id: 'review', label: 'Shot review', icon: Film, key: '3' },
    { id: 'settings', label: 'Settings', icon: SettingsIcon, key: '4' },
  ];
  let view = $state(location.hash.replace('#', '') || 'monitor');
  if (!views.some(v => v.id === view)) view = 'monitor';
  $effect(() => { location.hash = view; });

  const run = $derived(live.data?.run);
  const pill = $derived.by(() => {
    if (live.error) return { cls: 'pill-danger', text: 'backend offline' };
    if (!run) return { cls: '', text: 'no render yet' };
    const map = { running: ['pill-accent', `rendering · ${Math.round((run.progress ?? 0) * 100)}%`],
                  done: ['pill-ok', 'render done'], failed: ['pill-danger', 'render failed'],
                  stalled: ['pill-warn', 'render stalled'], idle: ['', 'idle'] };
    const [cls, text] = map[run.state] ?? ['', run.state];
    return { cls, text };
  });

  function onKey(e) {
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
    const v = views.find(v => v.key === e.key);
    if (v) view = v.id;
  }

  onMount(startPolling);
</script>

<svelte:window onkeydown={onKey} />

<div class="h-dvh grid grid-rows-[52px_1fr] md:grid-cols-[220px_1fr] md:grid-rows-[1fr]">
  <!-- rail -->
  <nav class="panel rounded-none border-0 border-b md:border-b-0 md:border-r flex md:flex-col items-center md:items-stretch px-2 md:px-3 md:py-4 gap-1 overflow-x-auto" aria-label="Views">
    <div class="hidden md:flex items-center gap-2 px-2 pb-4 mb-2 border-b border-line">
      <span class="grid place-items-center size-8 rounded-md bg-accent text-on-accent"><Clapperboard size={18} /></span>
      <div>
        <div class="font-semibold leading-tight">Render Studio</div>
        <div class="text-xs text-fg-dim leading-tight">cat-duet-recast</div>
      </div>
    </div>
    {#each views as v}
      <button class="flex items-center gap-3 px-3 h-11 rounded-md text-left transition-colors
                     {view === v.id ? 'bg-raised text-fg border border-line-strong' : 'text-fg-muted hover:text-fg hover:bg-raised/60 border border-transparent'}"
              onclick={() => view = v.id} aria-current={view === v.id ? 'page' : undefined}>
        <v.icon size={18} />
        <span class="whitespace-nowrap">{v.label}</span>
        <kbd class="ml-auto hidden md:inline text-[11px] text-fg-dim font-mono">{v.key}</kbd>
      </button>
    {/each}
    <div class="md:mt-auto flex items-center gap-2 px-1 md:px-2 md:pt-4 ml-auto md:ml-0">
      <span class="pill {pill.cls}">
        {#if run?.state === 'running'}<span class="size-2 rounded-full bg-current animate-pulse"></span>{/if}
        {pill.text}
      </span>
    </div>
  </nav>

  <main class="min-h-0 overflow-auto p-4 md:p-6">
    {#if view === 'monitor'}<Monitor />
    {:else if view === 'refs'}<References />
    {:else if view === 'review'}<Review />
    {:else}<Settings />{/if}
  </main>
</div>
