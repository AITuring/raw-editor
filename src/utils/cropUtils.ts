import type { Crop } from 'react-image-crop';
import type { Adjustments } from './adjustments';

export type CropConstraintTransform = Pick<
  Adjustments,
  | 'orientationSteps'
  | 'flipHorizontal'
  | 'flipVertical'
  | 'transformDistortion'
  | 'transformVertical'
  | 'transformHorizontal'
  | 'transformRotate'
  | 'transformAspect'
  | 'transformScale'
  | 'transformXOffset'
  | 'transformYOffset'
  | 'lensDistortionAmount'
  | 'lensDistortionEnabled'
  | 'lensDistortionParams'
  | 'sectionVisibility'
>;

export function getOrientedDimensions(
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
): { width: number; height: number } {
  const isSwapped = orientationSteps === 1 || orientationSteps === 3;
  return {
    width: isSwapped ? imageHeight : imageWidth,
    height: isSwapped ? imageWidth : imageHeight,
  };
}

export function calculateCenteredCrop(
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
  aspectRatio: number | null,
  rotation: number = 0,
  constrainToImage: boolean = true,
  transform?: CropConstraintTransform,
): Crop | null {
  if (!aspectRatio || aspectRatio <= 0) return null;

  const { width: W, height: H } = getOrientedDimensions(imageWidth, imageHeight, orientationSteps);

  const angle = constrainToImage ? Math.abs(rotation) : 0;
  const rad = ((angle % 180) * Math.PI) / 180;
  const sin = Math.sin(rad);
  const cos = Math.cos(rad);

  const h_c = Math.min(H / (aspectRatio * sin + cos), W / (aspectRatio * cos + sin));
  const w_c = aspectRatio * h_c;

  const candidate: Crop = {
    unit: 'px',
    x: Math.round((W - w_c) / 2),
    y: Math.round((H - h_c) / 2),
    width: Math.round(w_c),
    height: Math.round(h_c),
  };

  if (isCropWithinBounds(candidate, W, H, rotation, constrainToImage, transform)) return candidate;

  let lower = 0;
  let upper = 1;
  let best: Crop | null = null;
  for (let index = 0; index < 18; index += 1) {
    const factor = (lower + upper) / 2;
    const width = w_c * factor;
    const height = h_c * factor;
    const test: Crop = {
      unit: 'px',
      x: (W - width) / 2,
      y: (H - height) / 2,
      width,
      height,
    };
    if (isCropWithinBounds(test, W, H, rotation, constrainToImage, transform)) {
      best = test;
      lower = factor;
    } else {
      upper = factor;
    }
  }

  return best
    ? {
        ...best,
        x: Math.round(best.x),
        y: Math.round(best.y),
        width: Math.max(1, Math.floor(best.width)),
        height: Math.max(1, Math.floor(best.height)),
      }
    : null;
}

function computeGeometryAutoCropScale(transform: CropConstraintTransform, width: number, height: number): number {
  const geometryVisible = transform.sectionVisibility?.geometry ?? true;
  const opticsVisible = transform.sectionVisibility?.optics ?? transform.sectionVisibility?.details ?? true;
  const distortion = geometryVisible ? (transform.transformDistortion ?? 0) : 0;
  const params = opticsVisible && transform.lensDistortionEnabled ? transform.lensDistortionParams : null;
  const lensAmount = ((transform.lensDistortionAmount ?? 100) / 100) * 2.5;
  const manualK = (distortion / 100) * 2.5;
  if (!params && Math.abs(manualK) < 1e-5) return 1;

  const cx = width / 2;
  const cy = height / 2;
  const halfDiagonal = Math.hypot(cx, cy);
  let maximumScale = 1;
  const points = [
    [cx, 0],
    [cx, height],
    [0, cy],
    [width, cy],
    [0, 0],
    [width, 0],
    [0, height],
    [width, height],
  ];

  for (const [x, y] of points) {
    const dx = x - cx;
    const dy = y - cy;
    const radius = Math.hypot(dx, dy);
    if (radius < 1e-6) continue;
    let mappedX = dx;
    let mappedY = dy;

    if (params) {
      const normalizedRadius = radius / halfDiagonal;
      const radiusSquared = normalizedRadius * normalizedRadius;
      const distortedRadius =
        params.model === 1
          ? normalizedRadius *
            (params.k1 * radiusSquared * normalizedRadius +
              params.k2 * radiusSquared +
              params.k3 * normalizedRadius +
              (1 - params.k1 - params.k2 - params.k3))
          : normalizedRadius *
            (1 + params.k1 * radiusSquared + params.k2 * radiusSquared ** 2 + params.k3 * radiusSquared ** 3);
      const effectiveRadius = normalizedRadius + (distortedRadius - normalizedRadius) * lensAmount;
      const scale = effectiveRadius / normalizedRadius;
      mappedX *= scale;
      mappedY *= scale;
    }

    if (Math.abs(manualK) >= 1e-5) {
      const normalizedRadiusSquared = (mappedX * mappedX + mappedY * mappedY) / (cx * cx + cy * cy);
      const factor = 1 + manualK * normalizedRadiusSquared;
      mappedX *= factor;
      mappedY *= factor;
    }
    maximumScale = Math.max(maximumScale, Math.hypot(mappedX, mappedY) / radius);
  }

  return maximumScale > 1 ? maximumScale * 1.002 : maximumScale;
}

function isGeometrySampleVisible(
  orientedX: number,
  orientedY: number,
  orientedWidth: number,
  orientedHeight: number,
  sourceWidth: number,
  sourceHeight: number,
  rotation: number,
  transform: CropConstraintTransform,
): boolean {
  const orientedCenterX = orientedWidth / 2;
  const orientedCenterY = orientedHeight / 2;
  const rotationRadians = (-rotation * Math.PI) / 180;
  const rotationCosine = Math.cos(rotationRadians);
  const rotationSine = Math.sin(rotationRadians);
  const rotatedX =
    rotationCosine * (orientedX - orientedCenterX) - rotationSine * (orientedY - orientedCenterY) + orientedCenterX;
  const rotatedY =
    rotationSine * (orientedX - orientedCenterX) + rotationCosine * (orientedY - orientedCenterY) + orientedCenterY;

  const unflippedX = transform.flipHorizontal ? orientedWidth - rotatedX : rotatedX;
  const unflippedY = transform.flipVertical ? orientedHeight - rotatedY : rotatedY;
  const steps = ((transform.orientationSteps % 4) + 4) % 4;
  let geometryX = unflippedX;
  let geometryY = unflippedY;
  if (steps === 1) {
    geometryX = unflippedY;
    geometryY = sourceHeight - unflippedX;
  } else if (steps === 2) {
    geometryX = sourceWidth - unflippedX;
    geometryY = sourceHeight - unflippedY;
  } else if (steps === 3) {
    geometryX = sourceWidth - unflippedY;
    geometryY = unflippedX;
  }

  const geometryVisible = transform.sectionVisibility?.geometry ?? true;
  const vertical = geometryVisible ? (transform.transformVertical ?? 0) : 0;
  const horizontal = geometryVisible ? (transform.transformHorizontal ?? 0) : 0;
  const geometryRotation = geometryVisible ? (transform.transformRotate ?? 0) : 0;
  const aspect = geometryVisible ? (transform.transformAspect ?? 0) : 0;
  const scale = geometryVisible ? (transform.transformScale ?? 100) : 100;
  const xOffset = geometryVisible ? (transform.transformXOffset ?? 0) : 0;
  const yOffset = geometryVisible ? (transform.transformYOffset ?? 0) : 0;
  const centerX = sourceWidth / 2;
  const centerY = sourceHeight / 2;
  const offsetX = (xOffset / 100) * sourceWidth;
  const offsetY = (yOffset / 100) * sourceHeight;
  let x = geometryX - centerX - offsetX;
  let y = geometryY - centerY - offsetY;
  const perspectiveHorizontal = -horizontal / (50 * sourceWidth);
  const perspectiveVertical = vertical / (50 * sourceHeight);
  const perspectiveDenominator = 1 - perspectiveHorizontal * x - perspectiveVertical * y;
  if (!Number.isFinite(perspectiveDenominator) || Math.abs(perspectiveDenominator) < 1e-6) return false;
  x /= perspectiveDenominator;
  y /= perspectiveDenominator;

  const geometryRadians = (-geometryRotation * Math.PI) / 180;
  const geometryCosine = Math.cos(geometryRadians);
  const geometrySine = Math.sin(geometryRadians);
  const unrotatedX = geometryCosine * x - geometrySine * y;
  const unrotatedY = geometrySine * x + geometryCosine * y;
  const aspectFactor = aspect >= 0 ? 1 + aspect / 100 : 1 / (1 + Math.abs(aspect) / 100);
  const scaleFactor = scale / 100;
  if (Math.abs(scaleFactor) < 1e-6 || Math.abs(aspectFactor) < 1e-6) return false;
  let sourceX = centerX + unrotatedX / (scaleFactor * aspectFactor);
  let sourceY = centerY + unrotatedY / scaleFactor;

  const autoCropScale = computeGeometryAutoCropScale(transform, sourceWidth, sourceHeight);
  if (autoCropScale > 1) {
    sourceX = centerX + (sourceX - centerX) / autoCropScale;
    sourceY = centerY + (sourceY - centerY) / autoCropScale;
  }

  const opticsVisible = transform.sectionVisibility?.optics ?? transform.sectionVisibility?.details ?? true;
  const params = opticsVisible && transform.lensDistortionEnabled ? transform.lensDistortionParams : null;
  if (params) {
    const dx = sourceX - centerX;
    const dy = sourceY - centerY;
    const radius = Math.hypot(dx, dy);
    if (radius > 1e-6) {
      const halfDiagonal = Math.hypot(centerX, centerY);
      const normalizedRadius = radius / halfDiagonal;
      const radiusSquared = normalizedRadius * normalizedRadius;
      const distortedRadius =
        params.model === 1
          ? normalizedRadius *
            (params.k1 * radiusSquared * normalizedRadius +
              params.k2 * radiusSquared +
              params.k3 * normalizedRadius +
              (1 - params.k1 - params.k2 - params.k3))
          : normalizedRadius *
            (1 + params.k1 * radiusSquared + params.k2 * radiusSquared ** 2 + params.k3 * radiusSquared ** 3);
      const lensAmount = ((transform.lensDistortionAmount ?? 100) / 100) * 2.5;
      const effectiveRadius = normalizedRadius + (distortedRadius - normalizedRadius) * lensAmount;
      const radialScale = effectiveRadius / normalizedRadius;
      sourceX = centerX + dx * radialScale;
      sourceY = centerY + dy * radialScale;
    }
  }

  const manualDistortion = geometryVisible ? (transform.transformDistortion ?? 0) : 0;
  if (Math.abs(manualDistortion) >= 1e-5) {
    const dx = sourceX - centerX;
    const dy = sourceY - centerY;
    const normalizedRadiusSquared = (dx * dx + dy * dy) / (centerX * centerX + centerY * centerY);
    const factor = 1 + (manualDistortion / 100) * 2.5 * normalizedRadiusSquared;
    sourceX = centerX + dx * factor;
    sourceY = centerY + dy * factor;
  }

  return sourceX >= -0.5 && sourceY >= -0.5 && sourceX < sourceWidth - 0.5 && sourceY < sourceHeight - 0.5;
}

export function isCropWithinBounds(
  crop: Partial<Crop>,
  imageW: number,
  imageH: number,
  rotation: number,
  constrainToImage: boolean = true,
  transform?: CropConstraintTransform,
): boolean {
  if (
    crop.x === undefined ||
    crop.y === undefined ||
    !crop.width ||
    !crop.height ||
    crop.x < -1 ||
    crop.y < -1 ||
    crop.x + crop.width > imageW + 1 ||
    crop.y + crop.height > imageH + 1
  ) {
    return false;
  }

  if (!constrainToImage) {
    return true;
  }

  if (transform) {
    const steps = ((transform.orientationSteps % 4) + 4) % 4;
    const sourceWidth = steps === 1 || steps === 3 ? imageH : imageW;
    const sourceHeight = steps === 1 || steps === 3 ? imageW : imageH;
    const samples = [0, 0.25, 0.5, 0.75, 1];
    for (const xFactor of samples) {
      for (const yFactor of samples) {
        if (
          !isGeometrySampleVisible(
            crop.x + Math.max(0, crop.width - 1) * xFactor,
            crop.y + Math.max(0, crop.height - 1) * yFactor,
            imageW,
            imageH,
            sourceWidth,
            sourceHeight,
            rotation,
            transform,
          )
        ) {
          return false;
        }
      }
    }
    return true;
  }

  if (Math.abs(rotation) < 1e-6) return true;

  const cx = imageW / 2;
  const cy = imageH / 2;
  const rad = (-rotation * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const pts = [
    { x: crop.x, y: crop.y },
    { x: crop.x + crop.width, y: crop.y },
    { x: crop.x, y: crop.y + crop.height },
    { x: crop.x + crop.width, y: crop.y + crop.height },
  ];
  for (let i = 0; i < 4; i++) {
    const nx = cos * (pts[i].x - cx) - sin * (pts[i].y - cy) + cx;
    const ny = sin * (pts[i].x - cx) + cos * (pts[i].y - cy) + cy;
    if (nx < -1 || nx > imageW + 1 || ny < -1 || ny > imageH + 1) return false;
  }
  return true;
}

export function calculateAreaPreservingCrop(
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
  aspectRatio: number | null,
  rotation: number,
  currentCrop: Crop | null | undefined,
  constrainToImage: boolean = true,
  transform?: CropConstraintTransform,
): Crop | null {
  if (!aspectRatio || aspectRatio <= 0 || !currentCrop || !currentCrop.width || !currentCrop.height) return null;

  const { width: W, height: H } = getOrientedDimensions(imageWidth, imageHeight, orientationSteps);

  const area = currentCrop.width * currentCrop.height;
  const newH = Math.sqrt(area / aspectRatio);
  const newW = aspectRatio * newH;
  const centerX = currentCrop.x + currentCrop.width / 2;
  const centerY = currentCrop.y + currentCrop.height / 2;

  const candidate: Crop = {
    unit: 'px',
    x: Math.round(centerX - newW / 2),
    y: Math.round(centerY - newH / 2),
    width: Math.round(newW),
    height: Math.round(newH),
  };

  return isCropWithinBounds(candidate, W, H, rotation, constrainToImage, transform) ? candidate : null;
}

export function rotateCropCenter(
  crop: Crop,
  orientedWidth: number,
  orientedHeight: number,
  deltaDegrees: number,
): Crop {
  const rad = (deltaDegrees * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const cx = orientedWidth / 2;
  const cy = orientedHeight / 2;
  const px = crop.x + crop.width / 2 - cx;
  const py = crop.y + crop.height / 2 - cy;
  const rx = px * cos - py * sin;
  const ry = px * sin + py * cos;
  return {
    unit: 'px',
    x: Math.round(cx + rx - crop.width / 2),
    y: Math.round(cy + ry - crop.height / 2),
    width: crop.width,
    height: crop.height,
  };
}

export function rotateCropQuarterTurn(
  crop: Crop,
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
  direction: 1 | -1,
): Crop {
  const { width: currentWidth, height: currentHeight } = getOrientedDimensions(
    imageWidth,
    imageHeight,
    orientationSteps,
  );
  const pixelCrop =
    crop.unit === '%'
      ? {
          unit: 'px' as const,
          x: (crop.x / 100) * currentWidth,
          y: (crop.y / 100) * currentHeight,
          width: (crop.width / 100) * currentWidth,
          height: (crop.height / 100) * currentHeight,
        }
      : { ...crop, unit: 'px' as const };

  const rotated =
    direction === 1
      ? {
          x: currentHeight - pixelCrop.y - pixelCrop.height,
          y: pixelCrop.x,
        }
      : {
          x: pixelCrop.y,
          y: currentWidth - pixelCrop.x - pixelCrop.width,
        };

  return {
    unit: 'px',
    x: Math.round(rotated.x),
    y: Math.round(rotated.y),
    width: Math.round(pixelCrop.height),
    height: Math.round(pixelCrop.width),
  };
}
