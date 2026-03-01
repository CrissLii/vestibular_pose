export interface Candidate {
  action: string;
  label: string;
  score: number;
}

export interface RadarDataPoint {
  metric: string;
  score: number;
  level: string;
  value: number | null;
}

export interface CopPoint {
  x: number;
  y: number;
  t: number;
}

export interface SymmetryItem {
  label: string;
  left: number;
  right: number;
}

export interface EvalResponse {
  session_id: string;
  action_detected: string;
  action_detected_zh: string;
  candidates: Candidate[];
  metrics: Record<string, unknown>;
  grading: {
    pass: boolean;
    severity: string;
    reasons: Record<string, unknown>;
    suggestion?: string;
  };
  radar_data: RadarDataPoint[];
  cop_data: CopPoint[];
  symmetry_data: SymmetryItem[];
  annotated_video: string;
  report_url?: string;
}

export type AppPhase = 'idle' | 'uploading' | 'processing' | 'done' | 'error';
