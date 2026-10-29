export type BatchOutputFormat = 'jpg' | 'png' | 'tiff';
export type ProcessingPhase = 'idle' | 'analysis' | 'export';

export interface BatchGeometryModalProps {
  isOpen: boolean;
  onClose(): void;
}

export interface BatchGeometryProgress {
  current: number;
  path: string;
  phase?: 'analysis' | 'export';
  total: number;
}

export interface BatchGeometryPreviewItem {
  afterPreview: string;
  autoCorrected: boolean;
  beforePreview: string;
  contentConfidence: number;
  distortionApplied: boolean;
  orientationSteps: number;
  sourcePath: string;
  suggestedOrientationSteps: number;
}

export interface BatchGeometryAnalysisResult {
  items: BatchGeometryPreviewItem[];
}

export interface BatchGeometryCorrectionResult {
  contentOrientationCorrectedCount: number;
  distortionCorrectedCount: number;
  missingDistortionProfileCount: number;
  outputFolder: string;
  processedCount: number;
}
