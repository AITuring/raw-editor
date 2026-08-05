export type WhiteBalanceMode =
  'asShot' | 'auto' | 'daylight' | 'cloudy' | 'shade' | 'tungsten' | 'fluorescent' | 'flash' | 'custom';

interface WhiteBalancePreset {
  kelvin: number;
  tint: number;
}

type ExifData = Record<string, unknown> | null | undefined;

export const CAMERA_RAW_TINT_SCALE = 1.5;

const DEFAULT_AS_SHOT_KELVIN = 5500;
const MAX_MIRED_SHIFT = 150;
const MIN_KELVIN = 2000;
const MAX_KELVIN = 50000;

export const WHITE_BALANCE_PRESETS: Record<
  Exclude<WhiteBalanceMode, 'asShot' | 'auto' | 'custom'>,
  WhiteBalancePreset
> = {
  daylight: { kelvin: 5500, tint: 10 },
  cloudy: { kelvin: 6500, tint: 10 },
  shade: { kelvin: 7500, tint: 10 },
  tungsten: { kelvin: 2850, tint: 0 },
  fluorescent: { kelvin: 3800, tint: 21 },
  flash: { kelvin: 5500, tint: 10 },
};

const parseExifNumber = (value: unknown): number | null => {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string') return null;

  const match = value.replace(',', '.').match(/-?\d+(?:\.\d+)?/);
  if (!match) return null;

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
};

const lightSourceKelvin = (lightSource: number | null): number | null => {
  switch (lightSource) {
    case 1: // Daylight
    case 9: // Fine weather
      return 5500;
    case 2: // Fluorescent
      return 4000;
    case 3: // Tungsten
    case 24: // ISO studio tungsten
      return 2850;
    case 4: // Flash
      return 6000;
    case 10: // Cloudy
      return 6500;
    case 11: // Shade
      return 7500;
    case 12: // Daylight fluorescent
    case 13: // Day white fluorescent
    case 14: // Cool white fluorescent
    case 15: // White fluorescent
      return 5000;
    case 17: // Standard light A
      return 2850;
    case 18: // Standard light B
      return 4870;
    case 19: // Standard light C
      return 6770;
    case 20: // D55
      return 5500;
    case 21: // D65
      return 6500;
    case 22: // D75
      return 7500;
    case 23: // D50
      return 5000;
    default:
      return null;
  }
};

export const inferAsShotKelvin = (exif: ExifData): number => {
  const explicitTemperature = [exif?.AsShotTemperature, exif?.ColorTemperature, exif?.WhiteBalanceTemperature]
    .map(parseExifNumber)
    .find((value) => value !== null && value >= MIN_KELVIN && value <= MAX_KELVIN);

  if (explicitTemperature != null) return explicitTemperature;

  return lightSourceKelvin(parseExifNumber(exif?.LightSource)) ?? DEFAULT_AS_SHOT_KELVIN;
};

export const relativeTemperatureToKelvin = (temperature: number, asShotKelvin: number): number => {
  const safeAsShot = Math.max(MIN_KELVIN, Math.min(MAX_KELVIN, asShotKelvin || DEFAULT_AS_SHOT_KELVIN));
  const asShotMired = 1_000_000 / safeAsShot;
  const adjustedMired = asShotMired - (temperature / 100) * MAX_MIRED_SHIFT;
  if (adjustedMired <= 0) return MAX_KELVIN;
  return Math.max(MIN_KELVIN, Math.min(MAX_KELVIN, 1_000_000 / adjustedMired));
};

export const kelvinToRelativeTemperature = (kelvin: number, asShotKelvin: number): number => {
  const safeKelvin = Math.max(MIN_KELVIN, Math.min(MAX_KELVIN, kelvin));
  const safeAsShot = Math.max(MIN_KELVIN, Math.min(MAX_KELVIN, asShotKelvin || DEFAULT_AS_SHOT_KELVIN));
  const adjustedMired = 1_000_000 / safeKelvin;
  const asShotMired = 1_000_000 / safeAsShot;
  return Math.max(-100, Math.min(100, (-(adjustedMired - asShotMired) / MAX_MIRED_SHIFT) * 100));
};

export const cameraRawTintToRelative = (tint: number): number =>
  Math.max(-100, Math.min(100, tint / CAMERA_RAW_TINT_SCALE));

export const relativeTintToCameraRaw = (tint: number): number => tint * CAMERA_RAW_TINT_SCALE;
