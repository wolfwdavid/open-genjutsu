// Shared render status, polled every 3 s while the tab is visible. Used by the top bar and the Monitor view.
import { api } from './api.js';

export const live = $state({ data: null, error: null, selectedLog: null, lastOk: 0 });

let timer = null;

export async function refresh() {
  try {
    live.data = await api.status(live.selectedLog);
    live.error = null;
    live.lastOk = Date.now();
  } catch (e) {
    live.error = e.message;
  }
}

export function startPolling() {
  if (timer) return;
  refresh();
  timer = setInterval(() => { if (document.visibilityState === 'visible') refresh(); }, 3000);
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') refresh(); });
}

export function selectLog(name) {
  live.selectedLog = name;
  refresh();
}
