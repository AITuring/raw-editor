import { Adjustments, INITIAL_ADJUSTMENTS } from '../utils/adjustments';

/**
 * Compatibility model from the first browser-only RAW editor prototype.
 *
 * RapidRAW is the source of truth for decoding and rendering. Keeping this
 * small adapter lets us reuse the prototype's control vocabulary without
 * carrying its browser/WebGL decoder into the native app.
 */
export interface LegacyImageState {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  temperature: number;
  tint: number;
  saturation: number;
  vibrance: number;
  clarity: number;
  dehaze: number;
  sharpness: number;
  curve: Array<{ x: number; y: number }>;
}

export const LEGACY_DEFAULT_IMAGE_STATE: LegacyImageState = {
  exposure: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
  temperature: 5500,
  tint: 0,
  saturation: 0,
  vibrance: 0,
  clarity: 0,
  dehaze: 0,
  sharpness: 0,
  curve: [
    { x: 0, y: 0 },
    { x: 1, y: 1 },
  ],
};

const cloneAdjustments = (adjustments: Adjustments): Adjustments => structuredClone(adjustments);

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

/** Convert the prototype's 0..1 curve coordinates to RapidRAW's 0..255 form. */
const toRapidCurve = (curve: LegacyImageState['curve'], fallback: Array<{ x: number; y: number }>) => {
  const points = curve.length > 0 ? curve : fallback;
  return points
    .map((point) => ({
      x: clamp(Math.round(point.x * 255), 0, 255),
      y: clamp(Math.round(point.y * 255), 0, 255),
    }))
    .sort((a, b) => a.x - b.x);
};

/**
 * Map the original page's state into the native renderer's adjustment schema.
 * Temperature used 5500K as neutral in the prototype while RapidRAW stores a
 * signed correction, so neutral is explicitly normalized to zero here.
 */
export const legacyStateToAdjustments = (
  state: LegacyImageState,
  base: Adjustments = INITIAL_ADJUSTMENTS,
): Adjustments => {
  const next = cloneAdjustments(base);
  const curve = toRapidCurve(state.curve, LEGACY_DEFAULT_IMAGE_STATE.curve);

  return {
    ...next,
    exposure: state.exposure,
    contrast: state.contrast,
    highlights: state.highlights,
    shadows: state.shadows,
    whites: state.whites,
    blacks: state.blacks,
    temperature: clamp(Math.round((state.temperature - 5500) / 35), -100, 100),
    tint: state.tint,
    saturation: state.saturation,
    vibrance: state.vibrance,
    clarity: state.clarity,
    dehaze: state.dehaze,
    sharpness: state.sharpness,
    curves: {
      ...next.curves,
      luma: curve,
    },
    pointCurves: {
      ...(next.pointCurves || next.curves),
      luma: curve,
    },
  };
};

/** Convert native adjustments back to the old page's serializable state. */
export const adjustmentsToLegacyState = (adjustments: Adjustments): LegacyImageState => ({
  exposure: adjustments.exposure,
  contrast: adjustments.contrast,
  highlights: adjustments.highlights,
  shadows: adjustments.shadows,
  whites: adjustments.whites,
  blacks: adjustments.blacks,
  temperature: 5500 + adjustments.temperature * 35,
  tint: adjustments.tint,
  saturation: adjustments.saturation,
  vibrance: adjustments.vibrance,
  clarity: adjustments.clarity,
  dehaze: adjustments.dehaze,
  sharpness: adjustments.sharpness,
  curve: (adjustments.pointCurves?.luma || adjustments.curves.luma).map((point) => ({
    x: point.x / 255,
    y: point.y / 255,
  })),
});
