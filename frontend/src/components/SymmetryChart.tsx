import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { SymmetryItem } from '../types';

interface Props {
  data: SymmetryItem[];
}

export function SymmetryChart({ data }: Props) {
  if (data.length === 0) {
    return <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>无对称性数据</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
        <XAxis
          dataKey="label"
          tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(value: number | undefined) => (value != null ? value.toFixed(3) : '')}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, color: 'var(--text-muted)' }}
        />
        <Bar dataKey="left" name="左侧" fill="var(--primary)" radius={[4, 4, 0, 0]} />
        <Bar dataKey="right" name="右侧" fill="#22c55e" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
