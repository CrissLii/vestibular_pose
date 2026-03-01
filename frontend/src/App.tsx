import { useState, useCallback } from 'react';
import type { EvalResponse, AppPhase } from './types';
import { evaluateVideo, reEvaluate } from './api';
import { Header } from './components/Header';
import { UploadPanel } from './components/UploadPanel';
import { ResultPanel } from './components/ResultPanel';
import './App.css';

function App() {
  const [phase, setPhase] = useState<AppPhase>('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<EvalResponse | null>(null);
  const [error, setError] = useState('');
  const [view, setView] = useState('unknown');
  const [kptConf, setKptConf] = useState(0.20);

  const handleUpload = useCallback(async (file: File) => {
    setPhase('uploading');
    setProgress(0);
    setError('');
    setResult(null);

    try {
      setPhase('processing');
      const data = await evaluateVideo(file, view, kptConf, setProgress);
      setResult(data);
      setPhase('done');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('error');
    }
  }, [view, kptConf]);

  const handleReEvaluate = useCallback(async (actionId: string) => {
    if (!result) return;
    setPhase('processing');
    setProgress(50);
    try {
      const data = await reEvaluate(result.session_id, actionId);
      setResult(data);
      setPhase('done');
      setProgress(100);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('error');
    }
  }, [result]);

  const handleReset = useCallback(() => {
    setPhase('idle');
    setResult(null);
    setError('');
    setProgress(0);
  }, []);

  return (
    <div className="app">
      <Header />
      <main className="main-content">
        {phase !== 'done' ? (
          <UploadPanel
            phase={phase}
            progress={progress}
            error={error}
            view={view}
            kptConf={kptConf}
            onViewChange={setView}
            onKptConfChange={setKptConf}
            onUpload={handleUpload}
            onReset={handleReset}
          />
        ) : result && (
          <ResultPanel
            result={result}
            onReEvaluate={handleReEvaluate}
            onReset={handleReset}
          />
        )}
      </main>
    </div>
  );
}

export default App;
