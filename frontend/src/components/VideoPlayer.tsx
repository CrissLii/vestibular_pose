import { useRef, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import './VideoPlayer.css';

interface Props {
  src: string;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 1.5, 2];

export function VideoPlayer({ src }: Props) {
  const ref = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const toggle = () => {
    if (!ref.current) return;
    if (playing) {
      ref.current.pause();
    } else {
      ref.current.play();
    }
    setPlaying(!playing);
  };

  const skip = (dt: number) => {
    if (!ref.current) return;
    ref.current.currentTime = Math.max(0, ref.current.currentTime + dt);
  };

  const changeSpeed = () => {
    const idx = SPEED_OPTIONS.indexOf(speed);
    const next = SPEED_OPTIONS[(idx + 1) % SPEED_OPTIONS.length];
    setSpeed(next);
    if (ref.current) ref.current.playbackRate = next;
  };

  const fmt = (t: number) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="video-player">
      <video
        ref={ref}
        src={src}
        className="video-element"
        onTimeUpdate={() => setCurrentTime(ref.current?.currentTime || 0)}
        onLoadedMetadata={() => setDuration(ref.current?.duration || 0)}
        onEnded={() => setPlaying(false)}
        onClick={toggle}
      />
      <div className="video-controls">
        <button className="vc-btn" onClick={() => skip(-5)}>
          <SkipBack size={16} />
        </button>
        <button className="vc-btn vc-play" onClick={toggle}>
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button className="vc-btn" onClick={() => skip(5)}>
          <SkipForward size={16} />
        </button>

        <div className="vc-timeline">
          <input
            type="range"
            min={0}
            max={duration || 1}
            step={0.1}
            value={currentTime}
            onChange={(e) => {
              const t = parseFloat(e.target.value);
              if (ref.current) ref.current.currentTime = t;
              setCurrentTime(t);
            }}
          />
        </div>

        <span className="vc-time">{fmt(currentTime)} / {fmt(duration)}</span>

        <button className="vc-btn vc-speed" onClick={changeSpeed}>
          {speed}x
        </button>
      </div>
    </div>
  );
}
