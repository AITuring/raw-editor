export const BASE_RATIO = 1.618;

export const ORIGINAL_RATIO = 0;

export const RATIO_TOLERANCE = 0.01;

export function orientRatio(ratio: number, portrait: boolean): number {
  if (Math.abs(ratio - 1) < RATIO_TOLERANCE) return 1;
  if (portrait) return ratio > 1 ? 1 / ratio : ratio;
  return ratio < 1 ? 1 / ratio : ratio;
}

export interface CropPreset {
  id: string;
  name: string;
  value: number | null;
  tooltip: string;
}
