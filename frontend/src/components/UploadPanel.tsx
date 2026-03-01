import { useRef, useState, useCallback } from 'react';
import { Upload, Video, Settings, AlertCircle, Loader2 } from 'lucide-react';
import type { AppPhase } from '../types';
import './UploadPanel.css';

interface Props {
  phase: AppPhase;
  progress: number;
  error: string;
  view: string;
  kptConf: number;
  onViewChange: (v: string) => void;
  onKptConfChange: (v: number) => void;
  onUpload: (f: File) => void;
  onReset: () => void;
}

export function UploadPanel({
  phase, progress, error, view, kptConf,
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

  return (
    <div className="upload-panel">
      <div className="upload-card">
        {/* Drop zone */}
        <div
          className={`dropzone ${dragOver ? 'drag-over' : ''} ${selectedFile ? 'has-file' : ''}`}
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => !isProcessing && inputRef.current?.click()}
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

        {/* Settings */}
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

        {/* Action buttons */}
        <div className="action-row">
          <button
            className="btn btn-primary"
            disabled={!selectedFile || isProcessing}
            onClick={handleStart}
          >
            {isProcessing ? (
              <>
                <Loader2 size={16} className="spin" />
                分析中...
              </>
            ) : '开始评估'}
          </button>
          {(phase === 'error' || selectedFile) && (
            <button className="btn btn-ghost" onClick={() => { onReset(); setSelectedFile(null); }}>
              重置
            </button>
          )}
        </div>

        {/* Progress bar */}
        {isProcessing && (
          <div className="progress-container">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <span className="progress-text">
              {progress < 30 ? '正在上传视频...' :
               progress < 80 ? '正在进行姿态估计与动作评估...' :
               '正在生成标注视频...'}
            </span>
          </div>
        )}

        {/* Error */}
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
