import { useRef, useState, useCallback } from 'react';
import {
  Upload, Video, Settings, AlertCircle, Loader2,
  ScanLine, Brain, BarChart3, Film, CheckCircle2,
} from 'lucide-react';
import type { AppPhase } from '../types';
import type { StepDoneEvent } from '../api';
import './UploadPanel.css';

interface Props {
  phase: AppPhase;
  progress: number;
  stage: string;
  completedSteps: Record<string, StepDoneEvent>;
  error: string;
  view: string;
  kptConf: number;
  onViewChange: (v: string) => void;
  onKptConfChange: (v: number) => void;
  onUpload: (f: File) => void;
  onReset: () => void;
}

const PIPELINE_STEPS = [
  { id: 'pose',     label: '姿态估计',   icon: ScanLine },
  { id: 'detect',   label: '动作识别',   icon: Brain },
  { id: 'evaluate', label: '指标计算',   icon: BarChart3 },
  { id: 'render',   label: '视频渲染',   icon: Film },
  { id: 'done',     label: '完成',       icon: CheckCircle2 },
];

// Order index for determining status
const STEP_ORDER = ['pose', 'detect', 'evaluate', 'render', 'done'];

function getStatus(
  stepId: string,
  completedSteps: Record<string, StepDoneEvent>,
): 'pending' | 'active' | 'completed' {
  if (completedSteps[stepId]) return 'completed';

  const completedIds = Object.keys(completedSteps);
  if (completedIds.length === 0) {
    return stepId === 'pose' ? 'active' : 'pending';
  }

  // Find the max completed index
  let maxDone = -1;
  for (const cid of completedIds) {
    const idx = STEP_ORDER.indexOf(cid);
    if (idx > maxDone) maxDone = idx;
  }

  const myIdx = STEP_ORDER.indexOf(stepId);
  if (myIdx === maxDone + 1) return 'active';
  return 'pending';
}

export function UploadPanel({
  phase, progress, stage, completedSteps, error, view, kptConf,
  onViewChange, onKptConfChange, onUpload, onReset,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
    }
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setSelectedFile(file);
  }, []);

  const handleStart = () => {
    if (selectedFile) onUpload(selectedFile);
  };

  const isProcessing = phase === 'uploading' || phase === 'processing';

  // Processing view
  if (isProcessing) {
    return (
      <div className="upload-panel">
        <div className="processing-card">
          <div className="processing-header">
            <div className="pulse-ring">
              <Loader2 size={32} className="spin processing-icon" />
            </div>
            <h2 className="processing-title">正在分析视频</h2>
            <p className="processing-stage">{stage}</p>
          </div>

          {/* Pipeline steps with real timing */}
          <div className="pipeline-steps">
            {PIPELINE_STEPS.map((step, i) => {
              const status = getStatus(step.id, completedSteps);
              const Icon = step.icon;
              const doneEvt = completedSteps[step.id];
              return (
                <div key={step.id} className={`pipeline-step step-${status}`}>
                  <div className="step-icon-wrap">
                    {status === 'active' ? (
                      <Loader2 size={18} className="spin" />
                    ) : status === 'completed' ? (
                      <CheckCircle2 size={18} />
                    ) : (
                      <Icon size={18} />
                    )}
                  </div>
                  <span className="step-label">{step.label}</span>
                  {doneEvt && (
                    <span className="step-elapsed">{doneEvt.elapsed}s</span>
                  )}
                  {i < PIPELINE_STEPS.length - 1 && (
                    <div className={`step-connector ${status === 'completed' ? 'filled' : ''}`} />
                  )}
                </div>
              );
            })}
          </div>

          {/* Progress bar */}
          <div className="progress-section">
            <div className="progress-bar-large">
              <div className="progress-fill-large" style={{ width: `${progress}%` }}>
                <div className="progress-glow" />
              </div>
            </div>
            <div className="progress-meta">
              <span className="progress-pct">{Math.round(progress)}%</span>
              <span className="progress-hint">请耐心等待，视频越长处理时间越久</span>
            </div>
          </div>

          {selectedFile && (
            <div className="processing-file-info">
              <Video size={14} />
              <span>{selectedFile.name}</span>
              <span className="file-size-sm">
                ({(selectedFile.size / 1024 / 1024).toFixed(1)} MB)
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Idle / error view
  return (
    <div className="upload-panel">
      <div className="upload-card">
        <div
          className={`dropzone ${dragOver ? 'drag-over' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            onChange={handleFileInput}
            hidden
          />
          {selectedFile ? (
            <div className="file-preview">
              <Video size={40} className="file-icon" />
              <span className="file-name">{selectedFile.name}</span>
              <span className="file-size">
                {(selectedFile.size / 1024 / 1024).toFixed(1)} MB
              </span>
            </div>
          ) : (
            <div className="drop-hint">
              <Upload size={48} className="drop-icon" />
              <p className="drop-title">拖拽视频文件到此处</p>
              <p className="drop-sub">或点击选择文件 · 支持 MP4 / MOV / AVI</p>
            </div>
          )}
        </div>

        <div className="settings-row">
          <div className="setting">
            <Settings size={14} />
            <label>拍摄视角</label>
            <select value={view} onChange={(e) => onViewChange(e.target.value)}>
              <option value="unknown">自动识别</option>
              <option value="front">正面</option>
              <option value="side">侧面</option>
            </select>
          </div>
          <div className="setting">
            <label>关键点置信度</label>
            <input
              type="range"
              min={0.05}
              max={0.50}
              step={0.05}
              value={kptConf}
              onChange={(e) => onKptConfChange(parseFloat(e.target.value))}
            />
            <span className="conf-value">{kptConf.toFixed(2)}</span>
          </div>
        </div>

        <div className="action-row">
          <button
            className="btn btn-primary"
            disabled={!selectedFile}
            onClick={handleStart}
          >
            开始评估
          </button>
          {(phase === 'error' || selectedFile) && (
            <button className="btn btn-ghost" onClick={() => { onReset(); setSelectedFile(null); }}>
              重置
            </button>
          )}
        </div>

        {phase === 'error' && (
          <div className="error-banner">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
