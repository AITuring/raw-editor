/**
 * Parse an EXIF exposure-time value into seconds without changing the value
 * stored in the image metadata. EXIF readers may return either a decimal
 * value ("0.6 s") or a reciprocal fraction ("1/200 s").
 */
export function parseExposureTimeSeconds(value: unknown): number | null {
  if (typeof value === 'number') {
    return Number.isFinite(value) && value > 0 ? value : null;
  }

  if (typeof value !== 'string') return null;

  let cleaned = value.trim();
  let unitScale = 1;
  const milliseconds = cleaned.match(/^(.*?)\s*(?:milliseconds?|msec|ms)$/i);
  if (milliseconds) {
    cleaned = milliseconds[1]?.trim() || '';
    unitScale = 0.001;
  } else {
    cleaned = cleaned.replace(/\s*(?:seconds?|secs?|sec|s)\s*$/i, '').trim();
  }

  if (!cleaned) return null;

  const fraction = cleaned.match(/^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\/\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))$/);
  if (fraction) {
    const numerator = Number(fraction[1]);
    const denominator = Number(fraction[2]);
    if (!Number.isFinite(numerator) || !Number.isFinite(denominator) || denominator === 0) return null;

    const seconds = (numerator / denominator) * unitScale;
    return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
  }

  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(cleaned)) return null;

  const seconds = Number(cleaned) * unitScale;
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

function formatCompactNumber(value: number): string {
  // Two fractional places remove binary/float noise without inventing
  // excessive precision for uncommon intermediate shutter values.
  const rounded = Math.round(value * 100) / 100;
  if (!Number.isFinite(rounded)) return '';
  return rounded.toFixed(2).replace(/\.?0+$/, '');
}

/**
 * Format an exposure time for UI presentation. Values at or below 0.5 s use
 * the familiar reciprocal notation; slower exposures remain decimal seconds.
 * Invalid/unknown values are returned verbatim so metadata is never hidden.
 */
export function formatExposureTime(value: unknown): string {
  const raw = value == null ? '' : String(value).trim();
  if (!raw) return '';

  const seconds = parseExposureTimeSeconds(value);
  if (seconds === null) return raw;

  if (seconds >= 1) return `${formatCompactNumber(seconds)} s`;

  const denominator = 1 / seconds;
  if (denominator >= 2) return `1/${formatCompactNumber(denominator)} s`;

  return `${formatCompactNumber(seconds)} s`;
}
