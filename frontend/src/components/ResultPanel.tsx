import { useState } from 'react';
import { RotateCcw, Star, ChevronDown, ChevronUp, Download } from 'lucide-react';
import type { EvalResponse } from '../types';
import { RadarChart } from './RadarChart';
import { CopTrajectory } from './CopTrajectory';
import { SymmetryChart } from './SymmetryChart';
import { MetricTable } from './MetricTable';
import { VideoPlayer } from './VideoPlayer';
import { SeverityBadge, overallStars } from './SeverityBadge';
import './ResultPanel.css';

interface Props {
  result: EvalResponse;
  onReEvaluate: (actionId: string) => void;
  onReset: () => void;
}

const ACTION_OPTIONS = [
  { id: 'spin_in_place', label: '原地旋转' },
  { id: 'jump_in_place', label: '原地向上跳跃' },
  { id: 'wheelbarrow_walk', label: '小推车' },
  { id: 'run_straight', label: '直线加速跑' },
  { id: 'forward_roll', label: '前滚翻' },
  { id: 'head_up_prone', label: '抬头向上' },
];

export function ResultPanel({ result, onReEvaluate, onReset }: Props) {
  const [showCandidates, setShowCandidates] = useState(false);
  const [selectedAction, setSelectedAction] = useState(result.action_detected);

  const severity = result.grading.severity || '正常';
  const stars = overallStars(severity);

  const handleReEval = () => {
    if (selectedAction !== result.action_detected) {
      onReEvaluate(selectedAction);
    }
  };

  return (
    <div className="result-panel">
      {/* Top summary */}
      <div className="result-header">
        <div className="result-summary-card">
          <div className="summary-left">
            <h2 className="detected-action">{result.action_detected_zh}</h2>
            <div className="overall-row">
              <SeverityBadge level={severity} />
              <div className="stars">
                {Array.from({ length: 5 }, (_, i) => (
                  <Star
                    key={i}
                    size={20}
                    className={i < stars ? 'star-filled' : 'star-empty'}
                    fill={i < stars ? '#f59e0b' : 'none'}
                  />
                ))}
              </div>
            </div>
            {result.grading.suggestion && (
              <p className="suggestion">{result.grading.suggestion}</p>
            )}
          </div>
          <div className="summary-actions">
            <button className="btn btn-ghost" onClick={onReset}>
              <RotateCcw size={14} />
              重新上传
            </button>
            {result.report_url && (
              <a className="btn btn-ghost" href={result.report_url} download>
                <Download size={14} />
                下载报告
              </a>
            )}
          </div>
        </div>

        {/* Re-evaluate row */}
        <div className="re-eval-row">
          <span className="re-eval-label">手动选择动作：</span>
          <select
            value={selectedAction}
            onChange={(e) => setSelectedAction(e.target.value)}
            className="action-select"
          >
            {ACTION_OPTIONS.map((a) => (
              <option key={a.id} value={a.id}>{a.label}</option>
            ))}
          </select>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleReEval}
            disabled={selectedAction === result.action_detected}
          >
            重新评估
          </button>

          <button
            className="candidates-toggle"
            onClick={() => setShowCandidates(!showCandidates)}
          >
            候选动作
            {showCandidates ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {showCandidates && (
          <div className="candidates-list">
            {result.candidates.map((c) => (
              <div
                key={c.action}
                className={`candidate-item ${c.action === result.action_detected ? 'active' : ''}`}
                onClick={() => setSelectedAction(c.action)}
              >
                <span className="candidate-label">{c.label}</span>
                <div className="candidate-bar-bg">
                  <div
                    className="candidate-bar-fill"
                    style={{ width: `${Math.min(100, c.score * 100)}%` }}
                  />
                </div>
                <span className="candidate-score">{(c.score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Main content grid */}
      <div className="result-grid">
        {/* Left: Video */}
        <div className="result-col result-col-video">
          <div className="card">
            <h3 className="card-title">标注视频</h3>
            <VideoPlayer src={result.annotated_video} />
          </div>
        </div>

        {/* Right: Charts */}
        <div className="result-col result-col-charts">
          {result.radar_data.length > 0 && (
            <div className="card">
              <h3 className="card-title">评估雷达图</h3>
              <RadarChart data={result.radar_data} />
            </div>
          )}
          {result.cop_data.length > 0 && (
            <div className="card">
              <h3 className="card-title">重心轨迹 (COP)</h3>
              <CopTrajectory data={result.cop_data} />
            </div>
          )}
          {result.symmetry_data.length > 0 && (
            <div className="card">
              <h3 className="card-title">对称性分析</h3>
              <SymmetryChart data={result.symmetry_data} />
            </div>
          )}
        </div>
      </div>

      {/* Metric table */}
      <div className="card metric-section">
        <h3 className="card-title">详细指标</h3>
        <MetricTable reasons={result.grading.reasons} />
      </div>
    </div>
  );
}
