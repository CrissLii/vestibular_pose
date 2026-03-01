import './SeverityBadge.css';

const SEVERITY_MAP: Record<string, { label: string; cls: string }> = {
  '正常':     { label: '正常', cls: 'sev-normal' },
  '轻度偏差': { label: '轻度偏差', cls: 'sev-mild' },
  '中度偏差': { label: '中度偏差', cls: 'sev-moderate' },
  '重度偏差': { label: '重度偏差', cls: 'sev-severe' },
  NORMAL:    { label: '正常', cls: 'sev-normal' },
  MILD:      { label: '轻度偏差', cls: 'sev-mild' },
  MODERATE:  { label: '中度偏差', cls: 'sev-moderate' },
  SEVERE:    { label: '重度偏差', cls: 'sev-severe' },
};

export function SeverityBadge({ level }: { level: string }) {
  const info = SEVERITY_MAP[level] || SEVERITY_MAP['正常'];
  return <span className={`severity-badge ${info.cls}`}>{info.label}</span>;
}

export function overallStars(level: string): number {
  const stars: Record<string, number> = {
    '正常': 5, '轻度偏差': 4, '中度偏差': 2, '重度偏差': 1,
    NORMAL: 5, MILD: 4, MODERATE: 2, SEVERE: 1,
  };
  return stars[level] ?? 3;
}
