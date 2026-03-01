import type { EvalResponse } from './types';

const BASE = '';

export interface ProgressInfo {
  pct: number;
  stage: string;
}

const STAGES = [
  { at: 3,  label: '正在上传视频...' },
  { at: 8,  label: '视频上传完成，加载模型中...' },
  { at: 15, label: '正在进行姿态估计（YOLO Pose）...' },
  { at: 30, label: '提取人体关键点...' },
  { at: 45, label: '正在识别动作类型...' },
  { at: 55, label: '计算评估指标...' },
  { at: 68, label: '生成严重度分级...' },
  { at: 78, label: '渲染标注视频（骨骼叠加 + COP 轨迹）...' },
  { at: 88, label: '生成报告...' },
  { at: 93, label: '即将完成...' },
];

function stageFor(pct: number): string {
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (pct >= STAGES[i].at) return STAGES[i].label;
  }
  return STAGES[0].label;
}

export async function evaluateVideo(
  file: File,
  view: string,
  kptConf: number,
  onProgress?: (info: ProgressInfo) => void,
): Promise<EvalResponse> {
  const form = new FormData();
  form.append('video', file);
  form.append('view', view);
  form.append('kpt_conf', String(kptConf));

  let currentPct = 0;
  let done = false;

  // Simulated smooth progress: ramps up fast initially, slows near the end.
  // Stops at 94% and waits for the real response.
  const timer = setInterval(() => {
    if (done) return;
    const remaining = 94 - currentPct;
    const step = Math.max(0.3, remaining * 0.04);
    currentPct = Math.min(94, currentPct + step);
    onProgress?.({ pct: currentPct, stage: stageFor(currentPct) });
  }, 300);

  try {
    const resp = await fetch(`${BASE}/api/evaluate`, {
      method: 'POST',
      body: form,
    });

    done = true;
    clearInterval(timer);
    onProgress?.({ pct: 96, stage: '接收评估结果...' });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Evaluation failed');
    }

    const data: EvalResponse = await resp.json();
    onProgress?.({ pct: 100, stage: '评估完成！' });
    return data;
  } catch (e) {
    done = true;
    clearInterval(timer);
    throw e;
  }
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
