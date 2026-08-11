export const CROP_GUIDE_MODES = ['thirds', 'grid', 'diagonal', 'goldenTriangle', 'phiGrid', 'goldenSpiral'] as const;

export type VisibleCropGuideMode = (typeof CROP_GUIDE_MODES)[number];
export type CropGuideMode = VisibleCropGuideMode | 'none';

export const ROTATABLE_CROP_GUIDES = new Set<CropGuideMode>(['goldenTriangle', 'goldenSpiral']);

export const CROP_GUIDE_TRANSLATION_KEYS = {
  none: 'none',
  thirds: 'thirds',
  grid: 'grid',
  diagonal: 'diagonal',
  goldenTriangle: 'triangle',
  phiGrid: 'phiGrid',
  goldenSpiral: 'spiral',
} as const satisfies Record<CropGuideMode, string>;

export function getNextCropGuide(mode: CropGuideMode): VisibleCropGuideMode {
  const currentIndex = CROP_GUIDE_MODES.indexOf(mode as VisibleCropGuideMode);
  return CROP_GUIDE_MODES[(currentIndex + 1) % CROP_GUIDE_MODES.length];
}

export function getCropGuideOrientationCount(mode: CropGuideMode): number {
  if (mode === 'goldenSpiral') return 8;
  if (mode === 'goldenTriangle') return 2;
  return 1;
}
