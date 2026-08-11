import type { UprightGuide, UprightGuideAxis } from '../types/crop';

export interface UprightCorrection {
  rotation: number;
  vertical: number;
  horizontal: number;
  confidence: number;
}

interface Point {
  x: number;
  y: number;
}

const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));

const normalizeAxisResidual = (angle: number, axis: UprightGuideAxis) => {
  if (axis === 'horizontal') {
    let residual = angle;
    while (residual > 90) residual -= 180;
    while (residual <= -90) residual += 180;
    return residual;
  }

  let residual = angle - 90;
  while (residual > 90) residual -= 180;
  while (residual <= -90) residual += 180;
  return residual;
};

export function classifyUprightGuide(start: Point, end: Point, imageAspectRatio: number = 1): UprightGuideAxis {
  const aspectRatio = Number.isFinite(imageAspectRatio) && imageAspectRatio > 0 ? imageAspectRatio : 1;
  return Math.abs((end.x - start.x) * aspectRatio) >= Math.abs(end.y - start.y) ? 'horizontal' : 'vertical';
}

function rotatePoint(point: Point, degrees: number, aspectRatio: number): Point {
  const radians = (degrees * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const x = (point.x - 0.5) * aspectRatio;
  const y = point.y - 0.5;
  return {
    x: 0.5 + (x * cosine - y * sine) / aspectRatio,
    y: 0.5 + x * sine + y * cosine,
  };
}

function lineIntersection(left: UprightGuide, right: UprightGuide): Point | null {
  const x1 = left.start.x;
  const y1 = left.start.y;
  const x2 = left.end.x;
  const y2 = left.end.y;
  const x3 = right.start.x;
  const y3 = right.start.y;
  const x4 = right.end.x;
  const y4 = right.end.y;
  const denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

  if (Math.abs(denominator) < 1e-6) return null;

  const determinantLeft = x1 * y2 - y1 * x2;
  const determinantRight = x3 * y4 - y3 * x4;
  return {
    x: (determinantLeft * (x3 - x4) - (x1 - x2) * determinantRight) / denominator,
    y: (determinantLeft * (y3 - y4) - (y1 - y2) * determinantRight) / denominator,
  };
}

function strongestIntersection(guides: UprightGuide[], aspectRatio: number): Point | null {
  let best: { point: Point; score: number } | null = null;

  for (let leftIndex = 0; leftIndex < guides.length - 1; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < guides.length; rightIndex += 1) {
      const point = lineIntersection(guides[leftIndex], guides[rightIndex]);
      if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;

      const leftLength = Math.hypot(
        (guides[leftIndex].end.x - guides[leftIndex].start.x) * aspectRatio,
        guides[leftIndex].end.y - guides[leftIndex].start.y,
      );
      const rightLength = Math.hypot(
        (guides[rightIndex].end.x - guides[rightIndex].start.x) * aspectRatio,
        guides[rightIndex].end.y - guides[rightIndex].start.y,
      );
      const score = leftLength * rightLength;
      if (!best || score > best.score) best = { point, score };
    }
  }

  return best?.point ?? null;
}

function rotateGuide(guide: UprightGuide, degrees: number, aspectRatio: number): UprightGuide {
  return {
    ...guide,
    start: rotatePoint(guide.start, degrees, aspectRatio),
    end: rotatePoint(guide.end, degrees, aspectRatio),
  };
}

export function solveGuidedUpright(guides: UprightGuide[], imageAspectRatio: number = 1): UprightCorrection {
  if (guides.length < 2) {
    return { rotation: 0, vertical: 0, horizontal: 0, confidence: 0 };
  }

  const aspectRatio = Number.isFinite(imageAspectRatio) && imageAspectRatio > 0 ? imageAspectRatio : 1;

  let weightedResidual = 0;
  let totalLength = 0;
  for (const guide of guides) {
    const dx = (guide.end.x - guide.start.x) * aspectRatio;
    const dy = guide.end.y - guide.start.y;
    const length = Math.hypot(dx, dy);
    if (length < 1e-5) continue;
    const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
    weightedResidual += normalizeAxisResidual(angle, guide.axis) * length;
    totalLength += length;
  }

  const rotation = clamp(-(weightedResidual / Math.max(totalLength, 1e-6)), -15, 15);
  const rotatedGuides = guides.map((guide) => rotateGuide(guide, rotation, aspectRatio));
  const verticalGuides = rotatedGuides.filter((guide) => guide.axis === 'vertical');
  const horizontalGuides = rotatedGuides.filter((guide) => guide.axis === 'horizontal');

  const verticalVanishingPoint = strongestIntersection(verticalGuides, aspectRatio);
  const horizontalVanishingPoint = strongestIntersection(horizontalGuides, aspectRatio);

  const vertical = verticalVanishingPoint ? clamp(-50 / (verticalVanishingPoint.y - 0.5), -100, 100) : 0;
  const horizontal = horizontalVanishingPoint ? clamp(50 / (horizontalVanishingPoint.x - 0.5), -100, 100) : 0;
  const axisCoverage = Number(verticalGuides.length >= 2) + Number(horizontalGuides.length >= 2);
  const confidence = clamp(0.35 + guides.length * 0.1 + axisCoverage * 0.125, 0, 1);

  return {
    rotation: Math.round(rotation * 10) / 10,
    vertical: Math.round(vertical * 10) / 10,
    horizontal: Math.round(horizontal * 10) / 10,
    confidence,
  };
}

export function mapOrientedUprightCorrection(
  correction: UprightCorrection,
  orientationSteps: number,
  flipHorizontal: boolean,
  flipVertical: boolean,
): UprightCorrection {
  let { rotation, vertical, horizontal } = correction;

  if (flipHorizontal) {
    horizontal = -horizontal;
    rotation = -rotation;
  }
  if (flipVertical) {
    vertical = -vertical;
    rotation = -rotation;
  }

  const normalizedSteps = ((orientationSteps % 4) + 4) % 4;
  if (normalizedSteps === 1) {
    [vertical, horizontal] = [horizontal, -vertical];
  } else if (normalizedSteps === 2) {
    vertical = -vertical;
    horizontal = -horizontal;
  } else if (normalizedSteps === 3) {
    [vertical, horizontal] = [-horizontal, vertical];
  }

  return { ...correction, rotation, vertical, horizontal };
}

export function snapUprightGuides(guides: UprightGuide[]): UprightGuide[] {
  return guides.map((guide) => {
    const centerX = (guide.start.x + guide.end.x) / 2;
    const centerY = (guide.start.y + guide.end.y) / 2;
    const halfLength = Math.hypot(guide.end.x - guide.start.x, guide.end.y - guide.start.y) / 2;

    return guide.axis === 'horizontal'
      ? {
          ...guide,
          start: { x: clamp(centerX - halfLength, 0, 1), y: centerY },
          end: { x: clamp(centerX + halfLength, 0, 1), y: centerY },
        }
      : {
          ...guide,
          start: { x: centerX, y: clamp(centerY - halfLength, 0, 1) },
          end: { x: centerX, y: clamp(centerY + halfLength, 0, 1) },
        };
  });
}
