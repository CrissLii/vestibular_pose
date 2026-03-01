import type { EvalResponse } from './types';

const BASE = '';

export async function evaluateVideo(
  file: File,
  view: string,
  kptConf: number,
  onProgress?: (pct: number) => void,
): Promise<EvalResponse> {
  const form = new FormData();
  form.append('video', file);
  form.append('view', view);
  form.append('kpt_conf', String(kptConf));

  onProgress?.(10);

  const resp = await fetch(`${BASE}/api/evaluate`, {
    method: 'POST',
    body: form,
  });

  onProgress?.(90);

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'Evaluation failed');
  }

  const data: EvalResponse = await resp.json();
  onProgress?.(100);
  return data;
}

export async function reEvaluate(
  sessionId: string,
  actionId: string,
): Promise<EvalResponse> {
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('action_id', actionId);

  const resp = await fetch(`${BASE}/api/re-evaluate`, {
    method: 'POST',
    body: form,
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || 'Re-evaluation failed');
  }

  return resp.json();
}

export async function fetchActions(): Promise<{ id: string; label: string }[]> {
  const resp = await fetch(`${BASE}/api/actions`);
  return resp.json();
}
