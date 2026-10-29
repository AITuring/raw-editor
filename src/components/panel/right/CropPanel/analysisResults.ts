export interface StraightenAnalysisResult {
  angle: number;
  confidence: number;
  detected: boolean;
  lineCount: number;
}

export interface UprightAnalysisResult {
  rotation: number;
  vertical: number;
  horizontal: number;
  confidence: number;
  detected: boolean;
  lineCount: number;
}
