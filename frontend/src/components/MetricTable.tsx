import './MetricTable.css';

const LEVEL_LABELS: Record<string, string> = {
  NORMAL: '正常', MILD: '轻度偏差', MODERATE: '中度偏差', SEVERE: '重度偏差',
  '正常': '正常', '轻度偏差': '轻度偏差', '中度偏差': '中度偏差', '重度偏差': '重度偏差',
};

const LEVEL_CLASS: Record<string, string> = {
  NORMAL: 'normal', MILD: 'mild', MODERATE: 'moderate', SEVERE: 'severe',
  '正常': 'normal', '轻度偏差': 'mild', '中度偏差': 'moderate', '重度偏差': 'severe',
};

const METRIC_LABELS: Record<string, string> = {
  omega_avg: '平均旋转角速度',
  cv_omega: '角速度变异系数',
  d_head: '头部摆动',
  hang_time: '滞空时间',
  knee_angle: '膝关节着地角度',
  asym_limb: '左右肢体不对称',
  cop_range: 'COP 位移范围',
  trunk_drop: '躯干下沉角度',
  sl_sym: '步长对称性',
  ai_hand: '手部交替指数',
  cc_trunk: '躯干稳定性',
  lat_sway: '侧向摆动',
  peak_v: '峰值速度',
  accel_time: '加速时间',
  deviation: '跑偏距离',
  cadence: '步频',
  t_roll: '滚翻耗时',
  ang_roll: '滚翻角度覆盖',
  jerk: '动作平滑度 (jerk)',
  yaw: '偏航角',
  trunk_ang: '抬头躯干角度',
  hold_time: '保持时间',
  head_sway: '头部晃动',
  si_load: '承重对称指数',
};

interface Props {
  reasons: Record<string, unknown>;
}

export function MetricTable({ reasons }: Props) {
  const metrics: { key: string; value: unknown; level: string }[] = [];
  const keys = Object.keys(reasons);

  for (const k of keys) {
    if (k.endsWith('_level')) continue;
    const level = reasons[`${k}_level`];
    if (level !== undefined) {
      metrics.push({ key: k, value: reasons[k], level: String(level) });
    }
  }

  if (metrics.length === 0) {
    return <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>无详细指标数据</p>;
  }

  return (
    <table className="metric-table">
      <thead>
        <tr>
          <th>指标</th>
          <th>数值</th>
          <th>评级</th>
        </tr>
      </thead>
      <tbody>
        {metrics.map((m) => (
          <tr key={m.key}>
            <td className="mt-name">{METRIC_LABELS[m.key] || m.key}</td>
            <td className="mt-value">{formatValue(m.value)}</td>
            <td>
              <span className={`mt-level mt-level-${LEVEL_CLASS[m.level] || 'normal'}`}>
                {LEVEL_LABELS[m.level] || m.level}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return '—';
    return v % 1 === 0 ? String(v) : v.toFixed(3);
  }
  return String(v);
}
