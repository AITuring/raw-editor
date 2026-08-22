import { useCallback, useRef, type RefObject } from 'react';

import type { ScreenSpacePreviewGeometry } from '../utils/previewResolution';

interface ScreenSpacePreviewTransformOptions<Transform> {
  previewRef: RefObject<HTMLDivElement | null>;
  resolveGeometry(transform: Transform): ScreenSpacePreviewGeometry | null;
}

/**
 * Keeps an image preview responsive during zoom gestures without leaving the
 * settled bitmap behind a CSS scale transform. The live path transforms the
 * last baked screen-space rectangle; the settled path commits the new width,
 * height and position so WebKit resamples from the decoded source again.
 */
export function useScreenSpacePreviewTransform<Transform>({
  previewRef,
  resolveGeometry,
}: ScreenSpacePreviewTransformOptions<Transform>) {
  const bakedGeometryRef = useRef<ScreenSpacePreviewGeometry | null>(null);

  const bakePreview = useCallback(
    (transform: Transform) => {
      const element = previewRef.current;
      const geometry = resolveGeometry(transform);
      if (!element || !geometry) return;

      element.style.left = `${geometry.left}px`;
      element.style.top = `${geometry.top}px`;
      element.style.width = `${geometry.width}px`;
      element.style.height = `${geometry.height}px`;
      element.style.transform = 'none';
      element.style.transformOrigin = '0 0';
      bakedGeometryRef.current = geometry;
    },
    [previewRef, resolveGeometry],
  );

  const transformPreview = useCallback(
    (transform: Transform) => {
      const element = previewRef.current;
      const current = resolveGeometry(transform);
      const baked = bakedGeometryRef.current;
      if (!element || !current) return;

      if (!baked || baked.width <= 0 || baked.height <= 0) {
        bakePreview(transform);
        return;
      }

      const scaleX = current.width / baked.width;
      const scaleY = current.height / baked.height;
      const translateX = current.left - baked.left;
      const translateY = current.top - baked.top;
      element.style.transform = `matrix(${scaleX}, 0, 0, ${scaleY}, ${translateX}, ${translateY})`;
    },
    [bakePreview, previewRef, resolveGeometry],
  );

  const resetBakedPreview = useCallback(() => {
    bakedGeometryRef.current = null;
  }, []);

  return { bakePreview, resetBakedPreview, transformPreview };
}
