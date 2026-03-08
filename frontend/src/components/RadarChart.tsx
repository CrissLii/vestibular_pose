import {
  Radar,
  RadarChart as RChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { RadarDataPoint } from '../types';

const METRIC_LABELS: Record<string, string> = {
  omega_avg: '旋转速度',
  cv_omega: '速度稳定性',
  d_head: '头部稳定',
  hang_time: '滞空时间',
  knee_angle: '膝关节角度',
  asym_limb: '肢体对称',
  cop_range: 'COP范围',
  trunk_drop: '躯干高度',
  sl_sym: '步长对称',
  ai_hand: '手交替',
  cc_trunk: '躯干稳定',
  lat_sway: '侧向摆动',
  peak_v: '峰值速度',
  accel_time: '加速时间',
  deviation: '跑偏距离',
  cadence: '步频',
  t_roll: '滚翻时间',
  ang_roll: '滚翻角度',
  jerk: '动作平滑',
  yaw: '偏航角',
  trunk_ang: '躯干角度',
  hold_time: '保持时间',
  head_sway: '头部摆动',
  si_load: '承重对称',
};

interface Props {
  data: RadarDataPoint[];
}

export function RadarChart({ data }: Props) {
  const chartData = data.map((d) => ({
    ...d,
    label: METRIC_LABELS[d.metric] || d.metric,
    fullMark: 5,
  }));

  if (chartData.length < 3) {
    return <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>指标不足，无法生成雷达图</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RChart data={chartData}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis
          dataKey="label"
          tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
        />
        <PolarRadiusAxis
          domain={[0, 5]}
          tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
          axisLine={false}
        />
        <Radar
          dataKey="score"
          stroke="var(--primary)"
          fill="var(--primary)"
          fillOpacity={0.25}
          strokeWidth={2}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(value: number | undefined, _name: string, props: { payload: RadarDataPoint }) => {
            const p = props.payload;
            return value != null ? [`${value}/5 (${p.level})`, p.metric] : ['—', p.metric];
          }}
        />
      </RChart>
    </ResponsiveContainer>
  );
}
