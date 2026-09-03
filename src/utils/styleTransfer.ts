/**
 * Small, deterministic colour-transfer helpers used by the Style Lab.
 *
 * The previous one-off workflow in `new-chat` was a calibrated HSV pass
 * (`hue +10.4°`, lower saturation and a gentle value lift).  The UI needs to
 * work with arbitrary pairs, so we estimate those same parameters from the
 * reference and target previews instead of baking in a single image pair.
 * The functions intentionally operate on an ImageData-like shape so they can
 * be exercised without a DOM and reused by a worker in the future.
 */

export type StyleTransferMode = 'mood' | 'distribution';

export interface ImageDataLike {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface StyleTransferTransform {
  mode: StyleTransferMode;
  hueShift: number;
  saturationScale: number;
  valueScale: number;
  valueContrast: number;
  channelScale: [number, number, number];
  channelOffset: [number, number, number];
  targetMean: [number, number, number];
  referenceMean: [number, number, number];
}

export interface StyleTransferSummary {
  hueDegrees: number;
  saturationPercent: number;
  brightnessPercent: number;
  contrastPercent: number;
}

const clamp = (value: number, min = 0, max = 1) => Math.min(max, Math.max(min, value));

const normalizeHue = (value: number) => {
  let normalized = value;
  while (normalized > 0.5) normalized -= 1;
  while (normalized < -0.5) normalized += 1;
  return normalized;
};

const rgbToHsv = (r: number, g: number, b: number): [number, number, number] => {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let hue = 0;

  if (delta > 1e-6) {
    if (max === r) hue = ((g - b) / delta) % 6;
    else if (max === g) hue = (b - r) / delta + 2;
    else hue = (r - g) / delta + 4;
    hue /= 6;
    if (hue < 0) hue += 1;
  }

  return [hue, max <= 1e-6 ? 0 : delta / max, max];
};

const hsvToRgb = (hue: number, saturation: number, value: number): [number, number, number] => {
  const h = ((hue % 1) + 1) % 1;
  const scaled = h * 6;
  const sector = Math.floor(scaled);
  const fraction = scaled - sector;
  const p = value * (1 - saturation);
  const q = value * (1 - saturation * fraction);
  const t = value * (1 - saturation * (1 - fraction));

  switch (sector % 6) {
    case 0:
      return [value, t, p];
    case 1:
      return [q, value, p];
    case 2:
      return [p, value, t];
    case 3:
      return [p, q, value];
    case 4:
      return [t, p, value];
    default:
      return [value, p, q];
  }
};

interface ImageStats {
  count: number;
  mean: [number, number, number];
  std: [number, number, number];
  meanSaturation: number;
  meanValue: number;
  valueStd: number;
  hueSin: number;
  hueCos: number;
  hueWeight: number;
}

const collectStats = (image: ImageDataLike, maxSamples = 120_000): ImageStats => {
  const totalPixels = Math.max(1, image.width * image.height);
  const stride = Math.max(1, Math.ceil(Math.sqrt(totalPixels / maxSamples)));
  const data = image.data;
  const sum = [0, 0, 0];
  const sumSquares = [0, 0, 0];
  let saturationSum = 0;
  let valueSum = 0;
  let valueSquares = 0;
  let hueSin = 0;
  let hueCos = 0;
  let hueWeight = 0;
  let count = 0;

  for (let y = 0; y < image.height; y += stride) {
    for (let x = 0; x < image.width; x += stride) {
      const offset = (y * image.width + x) * 4;
      if (offset + 3 >= data.length) continue;
      const alpha = data[offset + 3] / 255;
      if (alpha < 0.05) continue;

      const r = data[offset] / 255;
      const g = data[offset + 1] / 255;
      const b = data[offset + 2] / 255;
      const [hue, saturation, value] = rgbToHsv(r, g, b);
      const weight = Math.max(0.001, saturation * 0.65 + value * 0.35) * alpha;

      sum[0] += r;
      sum[1] += g;
      sum[2] += b;
      sumSquares[0] += r * r;
      sumSquares[1] += g * g;
      sumSquares[2] += b * b;
      saturationSum += saturation;
      valueSum += value;
      valueSquares += value * value;
      hueSin += Math.sin(hue * Math.PI * 2) * weight;
      hueCos += Math.cos(hue * Math.PI * 2) * weight;
      hueWeight += weight;
      count += 1;
    }
  }

  const safeCount = Math.max(1, count);
  const mean: [number, number, number] = [sum[0] / safeCount, sum[1] / safeCount, sum[2] / safeCount];
  const std: [number, number, number] = [
    Math.sqrt(Math.max(0, sumSquares[0] / safeCount - mean[0] ** 2)),
    Math.sqrt(Math.max(0, sumSquares[1] / safeCount - mean[1] ** 2)),
    Math.sqrt(Math.max(0, sumSquares[2] / safeCount - mean[2] ** 2)),
  ];
  const meanValue = valueSum / safeCount;

  return {
    count,
    mean,
    std,
    meanSaturation: saturationSum / safeCount,
    meanValue,
    valueStd: Math.sqrt(Math.max(0, valueSquares / safeCount - meanValue ** 2)),
    hueSin,
    hueCos,
    hueWeight,
  };
};

/** Estimate a style transform from a reference image and a target image. */
export const analyzeStyleTransfer = (
  reference: ImageDataLike,
  target: ImageDataLike,
  mode: StyleTransferMode = 'mood',
): StyleTransferTransform => {
  const referenceStats = collectStats(reference);
  const targetStats = collectStats(target);

  const referenceHue =
    referenceStats.hueWeight > 0 ? Math.atan2(referenceStats.hueSin, referenceStats.hueCos) / (Math.PI * 2) : 0;
  const targetHue = targetStats.hueWeight > 0 ? Math.atan2(targetStats.hueSin, targetStats.hueCos) / (Math.PI * 2) : 0;

  const hueShift = normalizeHue(referenceHue - targetHue);
  const saturationScale = clamp(referenceStats.meanSaturation / Math.max(0.04, targetStats.meanSaturation), 0.35, 2.4);
  const valueScale = clamp(referenceStats.meanValue / Math.max(0.08, targetStats.meanValue), 0.55, 1.65);
  const valueContrast = clamp(referenceStats.valueStd / Math.max(0.04, targetStats.valueStd), 0.6, 1.7);
  const channelScale: [number, number, number] = [
    clamp(referenceStats.std[0] / Math.max(0.025, targetStats.std[0]), 0.55, 1.8),
    clamp(referenceStats.std[1] / Math.max(0.025, targetStats.std[1]), 0.55, 1.8),
    clamp(referenceStats.std[2] / Math.max(0.025, targetStats.std[2]), 0.55, 1.8),
  ];
  const channelOffset: [number, number, number] = [
    referenceStats.mean[0] - targetStats.mean[0] * channelScale[0],
    referenceStats.mean[1] - targetStats.mean[1] * channelScale[1],
    referenceStats.mean[2] - targetStats.mean[2] * channelScale[2],
  ];

  return {
    mode,
    hueShift,
    saturationScale,
    valueScale,
    valueContrast,
    channelScale,
    channelOffset,
    targetMean: targetStats.mean,
    referenceMean: referenceStats.mean,
  };
};

/** Apply an estimated transform in place, blending it by `strength` (0–1). */
export const applyStyleTransfer = (
  image: ImageDataLike,
  transform: StyleTransferTransform,
  strength: number,
): ImageDataLike => {
  const amount = clamp(strength);
  if (amount <= 0) return image;

  const data = image.data;
  const saturationScale = 1 + (transform.saturationScale - 1) * amount;
  const valueScale = 1 + (transform.valueScale - 1) * amount;
  const valueContrast = 1 + (transform.valueContrast - 1) * amount;
  const hueShift = transform.hueShift * amount;
  const channelScale: [number, number, number] = [
    1 + (transform.channelScale[0] - 1) * amount,
    1 + (transform.channelScale[1] - 1) * amount,
    1 + (transform.channelScale[2] - 1) * amount,
  ];
  const channelOffset: [number, number, number] = [
    transform.channelOffset[0] * amount,
    transform.channelOffset[1] * amount,
    transform.channelOffset[2] * amount,
  ];

  for (let offset = 0; offset < data.length; offset += 4) {
    if (data[offset + 3] < 13) continue;

    const originalR = data[offset] / 255;
    const originalG = data[offset + 1] / 255;
    const originalB = data[offset + 2] / 255;
    let nextR = originalR;
    let nextG = originalG;
    let nextB = originalB;

    if (transform.mode === 'distribution') {
      nextR = originalR * channelScale[0] + channelOffset[0];
      nextG = originalG * channelScale[1] + channelOffset[1];
      nextB = originalB * channelScale[2] + channelOffset[2];
    } else {
      const [hue, saturation, value] = rgbToHsv(originalR, originalG, originalB);
      const adjustedValue = clamp(
        transform.targetMean[0] * 0.2126 +
          transform.targetMean[1] * 0.7152 +
          transform.targetMean[2] * 0.0722 +
          (value -
            (transform.targetMean[0] * 0.2126 + transform.targetMean[1] * 0.7152 + transform.targetMean[2] * 0.0722)) *
            valueContrast,
      );
      [nextR, nextG, nextB] = hsvToRgb(
        hue + hueShift,
        clamp(saturation * saturationScale),
        clamp(adjustedValue * valueScale),
      );
    }

    data[offset] = Math.round(clamp(originalR + (nextR - originalR), 0, 1) * 255);
    data[offset + 1] = Math.round(clamp(originalG + (nextG - originalG), 0, 1) * 255);
    data[offset + 2] = Math.round(clamp(originalB + (nextB - originalB), 0, 1) * 255);
  }

  return image;
};

export const summarizeStyleTransform = (transform: StyleTransferTransform): StyleTransferSummary => ({
  hueDegrees: Math.round(transform.hueShift * 360),
  saturationPercent: Math.round((transform.saturationScale - 1) * 100),
  brightnessPercent: Math.round((transform.valueScale - 1) * 100),
  contrastPercent: Math.round((transform.valueContrast - 1) * 100),
});
