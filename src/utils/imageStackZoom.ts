export const FIT_TRANSFORM_SCALE = 1;
export const MAX_PIXEL_ZOOM = 8;
export const MAX_TRANSFORM_SCALE = 64;

const ZOOM_EPSILON = 0.0005;
const PIXEL_ZOOM_STOPS = [0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 6, 8] as const;

interface FitPixelZoomInput {
  devicePixelRatio: number;
  displayHeight: number;
  displayWidth: number;
  sourceHeight: number;
  sourceWidth: number;
}

export const calculateFitPixelZoom = ({
  devicePixelRatio,
  displayHeight,
  displayWidth,
  sourceHeight,
  sourceWidth,
}: FitPixelZoomInput): number => {
  if (
    !Number.isFinite(devicePixelRatio) ||
    !Number.isFinite(displayHeight) ||
    !Number.isFinite(displayWidth) ||
    !Number.isFinite(sourceHeight) ||
    !Number.isFinite(sourceWidth) ||
    devicePixelRatio <= 0 ||
    displayHeight <= 0 ||
    displayWidth <= 0 ||
    sourceHeight <= 0 ||
    sourceWidth <= 0
  ) {
    return 1;
  }

  return Math.min((displayWidth * devicePixelRatio) / sourceWidth, (displayHeight * devicePixelRatio) / sourceHeight);
};

export const calculatePixelZoom = (transformScale: number, fitPixelZoom: number): number =>
  transformScale * fitPixelZoom;

export const calculateMaxTransformScale = (fitPixelZoom: number): number => {
  if (!Number.isFinite(fitPixelZoom) || fitPixelZoom <= 0) return MAX_PIXEL_ZOOM;
  return Math.max(FIT_TRANSFORM_SCALE, Math.min(MAX_TRANSFORM_SCALE, MAX_PIXEL_ZOOM / fitPixelZoom));
};

export const resolveZoomStep = (
  currentTransformScale: number,
  direction: -1 | 1,
  fitPixelZoom: number,
  maxTransformScale: number,
): number => {
  const safeFitPixelZoom = Number.isFinite(fitPixelZoom) && fitPixelZoom > 0 ? fitPixelZoom : 1;
  const fitZoom = calculatePixelZoom(FIT_TRANSFORM_SCALE, safeFitPixelZoom);
  const currentZoom = calculatePixelZoom(currentTransformScale, safeFitPixelZoom);
  const maximumZoom = calculatePixelZoom(maxTransformScale, safeFitPixelZoom);

  let targetZoom = direction > 0 ? maximumZoom : fitZoom;
  if (direction > 0) {
    targetZoom = PIXEL_ZOOM_STOPS.find((stop) => stop > currentZoom + ZOOM_EPSILON) ?? maximumZoom;
  } else {
    for (let index = PIXEL_ZOOM_STOPS.length - 1; index >= 0; index -= 1) {
      const stop = PIXEL_ZOOM_STOPS[index];
      if (stop < currentZoom - ZOOM_EPSILON && stop >= fitZoom - ZOOM_EPSILON) {
        targetZoom = stop;
        break;
      }
    }
  }

  const clampedZoom = Math.min(maximumZoom, Math.max(fitZoom, targetZoom));
  return Math.min(maxTransformScale, Math.max(FIT_TRANSFORM_SCALE, clampedZoom / safeFitPixelZoom));
};

export const resolvePixelZoomScale = (pixelZoom: number, fitPixelZoom: number, maxTransformScale: number): number => {
  if (!Number.isFinite(fitPixelZoom) || fitPixelZoom <= 0) return FIT_TRANSFORM_SCALE;
  return Math.min(maxTransformScale, Math.max(FIT_TRANSFORM_SCALE, pixelZoom / fitPixelZoom));
};
