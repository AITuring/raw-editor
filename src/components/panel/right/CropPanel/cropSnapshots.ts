import { Adjustments, INITIAL_ADJUSTMENTS } from '../../../../utils/adjustments';

const CROP_SESSION_KEYS = [
  'aspectRatio',
  'constrainCrop',
  'crop',
  'flipHorizontal',
  'flipVertical',
  'orientationSteps',
  'rotation',
  'transformDistortion',
  'transformProjection',
  'transformVertical',
  'transformHorizontal',
  'transformRotate',
  'transformAspect',
  'transformScale',
  'transformXOffset',
  'transformYOffset',
] as const satisfies ReadonlyArray<keyof Adjustments>;

export type CropSessionSnapshot = Pick<Adjustments, (typeof CROP_SESSION_KEYS)[number]>;

const CROP_RESET_KEYS = [
  'aspectRatio',
  'constrainCrop',
  'crop',
  'flipHorizontal',
  'flipVertical',
  'orientationSteps',
  'rotation',
  'transformProjection',
  'transformVertical',
  'transformHorizontal',
  'transformRotate',
  'transformAspect',
  'transformScale',
  'transformXOffset',
  'transformYOffset',
] as const satisfies ReadonlyArray<keyof Adjustments>;

type CropResetSnapshot = Pick<Adjustments, (typeof CROP_RESET_KEYS)[number]>;

export function createCropSessionSnapshot(adjustments: Adjustments): CropSessionSnapshot {
  return Object.fromEntries(CROP_SESSION_KEYS.map((key) => [key, adjustments[key]])) as CropSessionSnapshot;
}

export function createCropResetSnapshot(originalAspectRatio: number | null): CropResetSnapshot {
  return {
    ...(Object.fromEntries(CROP_RESET_KEYS.map((key) => [key, INITIAL_ADJUSTMENTS[key]])) as CropResetSnapshot),
    aspectRatio: originalAspectRatio,
  };
}
