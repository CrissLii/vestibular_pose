import type { EvalResponse } from './types';

const BASE = '';

export interface StepDoneEvent {
  step: string;   // 'pose' | 'detect' | 'evaluate' | 'render' | 'done'
  elapsed: number; // seconds
  action_zh?: string;
}

export interface ProgressInfo {
  pct: number;
  stage: string;
  stepDone?: StepDoneEvent;
}

const STEP_PCT: Record<string, number> = {
  pose: 40,
  detect: 50,
  evaluate: 60,
  render: 85,
  done: 95,
};

const STEP_NEXT_STAGE: Record<string, string> = {
  pose: '正在识别动作类型...',
  detect: '计算评估指标...',
  evaluate: '渲染标注视频（骨骼叠加 + COP 轨迹）...',
  render: '生成图表与报告...',
  done: '即将完成...',
};

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

  onProgress?.({ pct: 2, stage: '正在上传视频...' });

  const resp = await fetch(`${BASE}/api/evaluate`, {
    method: 'POST',
    body: form,
  });

  if (!resp.ok && !resp.body) {
    throw new Error(`Server error: ${resp.status}`);
  }

  onProgress?.({ pct: 5, stage: '视频上传完成，开始姿态估计...' });

  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let result: EvalResponse | null = null;
  let currentPct = 5;

  // Smooth fill animation between steps
  let fillTarget = 5;
  let fillTimer: ReturnType<typeof setInterval> | null = null;

  const startFill = (target: number, label: string) => {
    fillTarget = target;
    if (fillTimer) clearInterval(fillTimer);
    fillTimer = setInterval(() => {
      if (currentPct < fillTarget - 1) {
        const remaining = fillTarget - currentPct;
        currentPct += Math.max(0.3, remaining * 0.06);
        onProgress?.({ pct: currentPct, stage: label });
      }
    }, 200);
  };

  startFill(38, '正在进行姿态估计（YOLO Pose）...');

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const evt = JSON.parse(line);

          if (evt.event === 'step_done') {
            const stepPct = STEP_PCT[evt.step] || currentPct;
            currentPct = stepPct;
            if (fillTimer) clearInterval(fillTimer);

            const nextStage = STEP_NEXT_STAGE[evt.step] || '处理中...';
            onProgress?.({
              pct: stepPct,
              stage: nextStage,
              stepDone: {
                step: evt.step,
                elapsed: evt.elapsed,
                action_zh: evt.action_zh,
              },
            });

            // Start smooth fill towards next step
            const nextStepPcts = Object.values(STEP_PCT).sort((a, b) => a - b);
            const nextTarget = nextStepPcts.find(p => p > stepPct) || 98;
            startFill(nextTarget, nextStage);

          } else if (evt.event === 'complete') {
            result = evt.result as EvalResponse;
            currentPct = 100;
            if (fillTimer) clearInterval(fillTimer);
            onProgress?.({ pct: 100, stage: '评估完成！' });

          } else if (evt.event === 'error') {
            if (fillTimer) clearInterval(fillTimer);
            throw new Error(evt.detail || 'Evaluation failed');
          }
        } catch (parseErr) {
          if (parseErr instanceof SyntaxError) continue;
          throw parseErr;
        }
      }
    }
  } finally {
    if (fillTimer) clearInterval(fillTimer);
  }

  if (!result) {
    throw new Error('No result received from server');
  }
  return result;
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
