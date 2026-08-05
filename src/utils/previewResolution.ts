export interface PreviewSize {
  width: number;
  height: number;
}

export interface PreviewResolutionOptions {
  displaySize: PreviewSize;
  baseRenderSize: PreviewSize;
  originalSize: PreviewSize;
  editorPreviewResolution?: number;
  enableZoomHifi?: boolean;
  useFullDpiRendering?: boolean;
  highResZoomMultiplier?: number;
  devicePixelRatio?: number;
}

export interface DevicePixelTranslationOptions {
  positionX: number;
  positionY: number;
  scale: number;
  imageOffsetX: number;
  imageOffsetY: number;
  devicePixelRatio?: number;
}

const MIN_PREVIEW_DIMENSION = 512;
const RESOLUTION_BUCKET = 256;
const FIT_ZOOM_EPSILON = 0.01;
const FULL_RESOLUTION_SNAP_RATIO = 0.8;
const ONE_TO_ONE_NEAR_RATIO = 0.9;
const SETTLED_SHARPNESS_FACTOR = 1.25;

const finitePositive = (value: number | undefined, fallback: number) =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;

const longestEdge = (size: PreviewSize) => Math.max(size.width || 0, size.height || 0);

/**
 * Resolves the longest edge requested from the native preview renderer.
 *
 * Fit-to-window rendering may honor the lower-DPI performance preference, but
 * once the user zooms beyond fit the settled preview must cover every physical
 * display pixel. This keeps 100% zoom at true source-pixel detail on Retina and
 * other high-DPI displays while CSS transforms remain instantaneous during the
 * gesture.
 */
export function calculatePreviewTargetResolution({
  displaySize,
  baseRenderSize,
  originalSize,
  editorPreviewResolution = 1920,
  enableZoomHifi = true,
  useFullDpiRendering = false,
  highResZoomMultiplier = 1,
  devicePixelRatio = 1,
}: PreviewResolutionOptions): number {
  const displayLongest = longestEdge(displaySize);
  const baseLongest = longestEdge(baseRenderSize);
  const sourceLongest = longestEdge(originalSize);
  const configuredBase = Math.max(MIN_PREVIEW_DIMENSION, finitePositive(editorPreviewResolution, 1920));

  if (!enableZoomHifi || displayLongest <= 0) {
    return Math.round(sourceLongest > 0 ? Math.min(configuredBase, sourceLongest) : configuredBase);
  }

  let target = configuredBase;

  const dpr = Math.max(1, finitePositive(devicePixelRatio, 1));
  const multiplier = finitePositive(highResZoomMultiplier, 1);
  const configuredDpr = useFullDpiRendering ? dpr : 1;

  target = Math.max(target, displayLongest * configuredDpr * SETTLED_SHARPNESS_FACTOR * multiplier);

  const isZoomedBeyondFit = baseLongest > 0 && displayLongest > baseLongest * (1 + FIT_ZOOM_EPSILON);
  const isNearSourcePixelScale = sourceLongest > 0 && displayLongest * dpr >= sourceLongest * ONE_TO_ONE_NEAR_RATIO;

  if (isZoomedBeyondFit || isNearSourcePixelScale) {
    // Quality multipliers may add oversampling, but must never reduce a
    // settled zoomed preview below one preview pixel per device pixel.
    target = Math.max(target, displayLongest * dpr);
  }

  if (sourceLongest > 0) {
    target = Math.min(target, sourceLongest);
    if (target >= sourceLongest * FULL_RESOLUTION_SNAP_RATIO) {
      return Math.round(sourceLongest);
    }
  }

  return Math.ceil(target / RESOLUTION_BUCKET) * RESOLUTION_BUCKET;
}

/** Aligns the rendered image origin, rather than the outer transform layer, to
 * the device pixel grid. Intended for settled integer pixel scales such as
 * 100% and 200%, where a half-device-pixel offset would visibly soften detail.
 */
export function snapImageTranslationToDevicePixels({
  positionX,
  positionY,
  scale,
  imageOffsetX,
  imageOffsetY,
  devicePixelRatio = 1,
}: DevicePixelTranslationOptions): { positionX: number; positionY: number } {
  const dpr = Math.max(1, finitePositive(devicePixelRatio, 1));
  const imageOriginX = positionX + imageOffsetX * scale;
  const imageOriginY = positionY + imageOffsetY * scale;
  const snappedOriginX = Math.round(imageOriginX * dpr) / dpr;
  const snappedOriginY = Math.round(imageOriginY * dpr) / dpr;

  return {
    positionX: positionX + snappedOriginX - imageOriginX,
    positionY: positionY + snappedOriginY - imageOriginY,
  };
}
