import { ActiveChannel, Coord, ParametricCurveSettings } from '../../../utils/adjustments';

export const DEFAULT_PARAMETRIC_CURVE_SETTINGS: ParametricCurveSettings = {
  darks: 0,
  shadows: 0,
  highlights: 0,
  lights: 0,
  whiteLevel: 0,
  blackLevel: 0,
  split1: 25,
  split2: 50,
  split3: 75,
};

export const DEFAULT_PARAMETRIC_CURVE = {
  luma: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
  red: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
  green: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
  blue: { ...DEFAULT_PARAMETRIC_CURVE_SETTINGS },
};

export function buildParametricPoints(settings: ParametricCurveSettings): Array<Coord> {
  const vH = settings.highlights / 100;
  const vL = settings.lights / 100;
  const vD = settings.darks / 100;
  const vS = settings.shadows / 100;

  const blackYOffset = settings.blackLevel;
  const whiteYOffset = settings.whiteLevel;

  const s1 = settings.split1 / 100;
  const s2 = settings.split2 / 100;
  const s3 = settings.split3 / 100;

  const xH = (s3 + 1) / 2;
  const xS = s1 / 2;
  const xs = [0, xS, s1, s2, s3, xH, 1];

  const SLIDER_GAIN = 1.2;
  const MAX_DISPLACEMENT = 0.35;

  const response = (v: number, x: number): number => {
    const headroom = v >= 0 ? 1 - x : x;
    const compressedHeadroom = Math.sqrt(headroom);
    const sigmoid = Math.tanh(v * SLIDER_GAIN);
    return sigmoid * MAX_DISPLACEMENT * compressedHeadroom;
  };

  const ys = [
    0,
    xS + response(vS, xS),
    s1 + (response(vS, s1) + response(vD, s1)) / 2,
    s2 + (response(vD, s2) + response(vL, s2)) / 2,
    s3 + (response(vL, s3) + response(vH, s3)) / 2,
    xH + response(vH, xH),
    1,
  ];

  const clamp = (v: number) => Math.max(0, Math.min(1, v));

  const points = xs.map((x, i) => ({
    x: x * 255,
    y: clamp(ys[i]) * 255,
  }));

  if (points.length >= 2) {
    points[0].y = Math.max(0, Math.min(255, points[0].y + blackYOffset));

    const lastIndex = points.length - 1;
    points[lastIndex].y = Math.max(0, Math.min(255, points[lastIndex].y + whiteYOffset));
  }

  return points;
}

export function isDefaultParametricCurve(settings: ParametricCurveSettings | undefined) {
  if (!settings) return true;
  return (
    settings.darks === DEFAULT_PARAMETRIC_CURVE_SETTINGS.darks &&
    settings.shadows === DEFAULT_PARAMETRIC_CURVE_SETTINGS.shadows &&
    settings.lights === DEFAULT_PARAMETRIC_CURVE_SETTINGS.lights &&
    settings.highlights === DEFAULT_PARAMETRIC_CURVE_SETTINGS.highlights &&
    settings.whiteLevel === DEFAULT_PARAMETRIC_CURVE_SETTINGS.whiteLevel &&
    settings.blackLevel === DEFAULT_PARAMETRIC_CURVE_SETTINGS.blackLevel &&
    settings.split1 === DEFAULT_PARAMETRIC_CURVE_SETTINGS.split1 &&
    settings.split2 === DEFAULT_PARAMETRIC_CURVE_SETTINGS.split2 &&
    settings.split3 === DEFAULT_PARAMETRIC_CURVE_SETTINGS.split3
  );
}

export function getSplitterGradient(channel: ActiveChannel) {
  switch (channel) {
    case ActiveChannel.Luma:
      return 'linear-gradient(to right, rgba(0, 0, 0, 0.8) 0%, rgba(64, 64, 64, 0.8) 25%, rgba(105, 101, 101, 0.8) 50%, rgba(158, 154, 154, 0.8) 75%, rgba(198, 195, 197, 0.8) 100%)';
    case ActiveChannel.Red:
      return 'linear-gradient(to right, rgba(0, 0, 0, 0.8) 0%, rgba(64, 0, 0, 0.8) 25%, rgba(105, 50, 50, 0.8) 50%, rgba(158, 100, 100, 0.8) 75%, rgba(255, 107, 107, 0.8) 100%)';
    case ActiveChannel.Green:
      return 'linear-gradient(to right, rgba(0, 0, 0, 0.8) 0%, rgba(0, 64, 0, 0.8) 25%, rgba(50, 105, 50, 0.8) 50%, rgba(100, 158, 100, 0.8) 75%, rgba(107, 203, 119, 0.8) 100%)';
    case ActiveChannel.Blue:
      return 'linear-gradient(to right, rgba(0, 0, 0, 0.8) 0%, rgba(0, 0, 64, 0.8) 25%, rgba(50, 50, 105, 0.8) 50%, rgba(100, 100, 158, 0.8) 75%, rgba(77, 150, 255, 0.8) 100%)';
    default:
      return 'linear-gradient(to right, rgba(0, 0, 0, 0.8) 0%, rgba(64, 64, 64, 0.8) 25%, rgba(105, 101, 101, 0.8) 50%, rgba(158, 154, 154, 0.8) 75%, rgba(198, 195, 197, 0.8) 100%)';
  }
}

export function convertParametricToPoints(settings: ParametricCurveSettings): Array<Coord> {
  return buildParametricPoints(settings);
}
