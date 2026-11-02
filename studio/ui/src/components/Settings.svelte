<script>
  import { onMount } from 'svelte';
  import { Save, Plug, CheckCircle2, AlertTriangle, Eye, EyeOff } from 'lucide-svelte';
  import { api } from '../lib/api.js';

  let s = $state({ editor_url: '', project_prefix: '', has_cookie: false });
  let cookie = $state('');
  let showCookie = $state(false);
  let saving = $state(false);
  let saved = $state(false);
  let ping = $state(null);
  let error = $state(null);

  onMount(async () => { try { s = await api.settings(); } catch (e) { error = e.message; } });

  async function save() {
    saving = true; error = null; saved = false;
    try {
      const body = { editor_url: s.editor_url, project_prefix: s.project_prefix };
      if (cookie) body.editor_cookie = cookie;
      s = await api.saveSettings(body);
      cookie = ''; saved = true; setTimeout(() => saved = false, 3000);
    } catch (e) { error = e.message; } finally { saving = false; }
  }
  async function test() { ping = null; try { ping = await api.ping(); } catch (e) { error = e.message; } }
</script>

<div class="max-w-2xl mx-auto grid gap-4">
  <header>
    <h1 class="text-xl font-semibold">Settings</h1>
    <p class="text-sm text-fg-muted">The "Send to editor" handoff to video_editor_ai. Stored in studio/settings.json (not committed).</p>
  </header>
  {#if error}<div class="panel p-3 border-danger/50 text-danger flex gap-2"><AlertTriangle size={18} /> {error}</div>{/if}

  <section class="panel p-5 grid gap-4">
    <label class="grid gap-1 text-sm"><span class="label">Editor URL</span>
      <input class="input font-mono" bind:value={s.editor_url} placeholder="http://127.0.0.1:8000" /></label>
    <label class="grid gap-1 text-sm"><span class="label">Project name prefix</span>
      <input class="input" bind:value={s.project_prefix} placeholder="Cat duet" /></label>
    <div class="grid gap-1 text-sm">
      <label class="label" for="cookie">Editor session cookie (sb_access_token)</label>
      <div class="flex gap-2">
        <input id="cookie" class="input font-mono" type={showCookie ? 'text' : 'password'} bind:value={cookie} placeholder={s.has_cookie ? 'saved · paste to replace' : 'paste the sb_access_token value'} autocomplete="off" />
        <button class="btn !px-2" aria-label={showCookie ? 'Hide cookie' : 'Show cookie'} onclick={() => showCookie = !showCookie}>{#if showCookie}<EyeOff size={16} />{:else}<Eye size={16} />{/if}</button>
      </div>
      <p class="text-fg-dim">The editor only accepts its own login session, so the Studio reuses yours: sign in to the editor, open DevTools → Application → Cookies, copy <code class="font-mono">sb_access_token</code>. It expires with the session; the handoff falls back to staging the file when it does.</p>
    </div>
    <div class="flex items-center gap-3">
      <button class="btn btn-primary" onclick={save} disabled={saving}><Save size={16} /> {saving ? 'Saving…' : 'Save'}</button>
      <button class="btn" onclick={test}><Plug size={16} /> Test connection</button>
      {#if saved}<span class="text-ok text-sm flex items-center gap-1"><CheckCircle2 size={16} /> saved</span>{/if}
    </div>
    {#if ping}
      <div class="card p-3 text-sm grid gap-1">
        <div class="flex items-center gap-2">{#if ping.reachable}<CheckCircle2 class="text-ok" size={16} /> Editor reachable at {ping.url} (HTTP {ping.status}){:else}<AlertTriangle class="text-danger" size={16} /> Not reachable: {ping.error}{/if}</div>
        {#if ping.reachable}
          <div class="flex items-center gap-2">
            {#if ping.session_ok === true}<CheckCircle2 class="text-ok" size={16} /> Session accepted
            {:else if ping.session_ok === false}<AlertTriangle class="text-warn" size={16} /> Session rejected (expired or wrong value)
            {:else}<AlertTriangle class="text-fg-dim" size={16} /> No session saved: the handoff will stage files for manual upload{/if}
          </div>
        {/if}
      </div>
    {/if}
  </section>

  <section class="panel p-5 grid gap-2 text-sm text-fg-muted">
    <div class="label">Starting the editor</div>
    <pre class="card p-3 font-mono text-[12.5px] overflow-auto">cd <path to video_editor_ai>
python run_app.py          # serves http://127.0.0.1:8000 (needs its Supabase stack for login)</pre>
  </section>
</div>
