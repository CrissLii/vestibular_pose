import { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type { CopPoint } from '../types';

interface Props {
  data: CopPoint[];
}

function lerp(a: string, b: string, t: number): string {
  const parse = (c: string) => {
    const m = c.match(/\w{2}/g)!;
    return m.map((x) => parseInt(x, 16));
  };
  const ca = parse(a), cb = parse(b);
  const r = ca.map((v, i) => Math.round(v + (cb[i] - v) * t));
  return `rgb(${r[0]},${r[1]},${r[2]})`;
}

export function CopTrajectory({ data }: Props) {
  const plotData = useMemo(() => {
    if (data.length === 0) return [];
    const minX = Math.min(...data.map((d) => d.x));
    const minY = Math.min(...data.map((d) => d.y));
    return data.map((d) => ({
      x: d.x - minX,
      y: d.y - minY,
      t: d.t,
    }));
  }, [data]);

  if (plotData.length < 2) {
    return <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>COP 数据不足</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          name="X"
          tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
          label={{ value: 'X (px)', position: 'bottom', fill: 'var(--text-muted)', fontSize: 11 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          name="Y"
          tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
          label={{ value: 'Y (px)', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 11 }}
          reversed
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            fontSize: 12,
          }}
          formatter={(value: number | undefined) => (value != null ? value.toFixed(1) : '')}
        />
        <Scatter data={plotData}>
          {plotData.map((entry, i) => (
            <Cell
              key={i}
              fill={lerp('6366f1', 'ef4444', entry.t)}
              opacity={0.7}
              r={3}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
