// Thin fetch wrapper for the Studio backend. Every call returns parsed JSON or throws Error(detail).
async function call(method, url, body, isForm = false) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    if (isForm) opts.body = body;
    else { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  }
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* keep statusText */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return r.json();
}

export const api = {
  status: (log) => call('GET', '/api/status' + (log ? `?log=${encodeURIComponent(log)}` : '')),
  refs: () => call('GET', '/api/refs'),
  uploadRefs: (files) => { const fd = new FormData(); for (const f of files) fd.append('files', f); return call('POST', '/api/refs/upload', fd, true); },
  deleteUpload: (name) => call('DELETE', `/api/refs/upload/${encodeURIComponent(name)}`),
  buildRef: () => call('POST', '/api/refs/build'),
  promoteRef: () => call('POST', '/api/refs/promote'),
  job: (id) => call('GET', `/api/jobs/${id}`),
  shots: () => call('GET', '/api/shots'),
  review: (n, body) => call('POST', `/api/shots/${n}/review`, body),
  videos: () => call('GET', '/api/videos'),
  settings: () => call('GET', '/api/settings'),
  saveSettings: (body) => call('POST', '/api/settings', body),
  ping: () => call('GET', '/api/handoff/ping'),
  handoff: (body) => call('POST', '/api/handoff', body),
};

export const fileUrl = (path) => `/api/file?path=${encodeURIComponent(path)}`;
export const frameUrl = (lane, i, w = 480) => `/api/frame/${lane}/${i}?w=${w}`;

export function fmtBytes(b) {
  if (b == null) return '';
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

export function fmtClock(epoch) {
  if (!epoch) return '';
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** cut frame (24 fps) -> mm:ss:ff timecode */
export function tc(cutFrame, fps = 24) {
  const s = Math.floor(cutFrame / fps), f = cutFrame % fps;
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}:${String(f).padStart(2, '0')}`;
}
