import { forwardRef, memo, useCallback, useEffect, useImperativeHandle, useRef, type ReactNode } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import { ScanSearch, ZoomIn, ZoomOut } from 'lucide-react';
import { useScreenSpacePreviewTransform } from '../../hooks/useScreenSpacePreviewTransform';
import { calculateScreenSpacePreviewGeometry } from '../../utils/previewResolution';
import ScreenSpacePreview from './ScreenSpacePreview';
import {
  FIT_TRANSFORM_SCALE,
  MAX_PIXEL_ZOOM,
  calculateFitPixelZoom,
  calculateMaxTransformScale,
  calculatePixelZoom,
  resolvePixelZoomScale,
  resolveZoomStep,
} from './imagePreviewZoom';
import './ZoomableImagePreview.css';

export interface ZoomableImagePreviewLabels {
  fit: string;
  toolbar?: string;
  zoomIn: string;
  zoomOut: string;
}

export interface ZoomableImagePreviewSourceSize {
  height: number;
  width: number;
}

export interface ZoomableImagePreviewHandle {
  fitToViewport(): void;
  zoomIn(): void;
  zoomOut(): void;
  zoomToActualSize(): void;
}

export interface ZoomableImagePreviewProps {
  alt: string;
  ariaDescribedBy?: string;
  children?: ReactNode;
  className?: string;
  detailSrc?: string | null;
  interactionSrc: string;
  labels: ZoomableImagePreviewLabels;
  sourceSize?: ZoomableImagePreviewSourceSize | null;
  toolbarEnd?: ReactNode;
}

interface PreviewTransform {
  scale: number;
  x: number;
  y: number;
}

interface PreviewGeometry {
  imageHeight: number;
  imageWidth: number;
  viewportHeight: number;
  viewportWidth: number;
}

const TRANSFORM_SETTLE_DELAY_MS = 200;
const TRANSFORM_ANIMATION_MS = 160;
const ZOOM_EPSILON = 0.001;
const IDENTITY_TRANSFORM: PreviewTransform = { scale: FIT_TRANSFORM_SCALE, x: 0, y: 0 };

const clampTransformScale = (scale: number, maxScale: number) =>
  Math.min(maxScale, Math.max(FIT_TRANSFORM_SCALE, scale));

const resolvePreviewCursor = (activePointer: number | null, scale: number, maxScale: number) => {
  if (activePointer !== null) return 'grabbing';
  if (scale > FIT_TRANSFORM_SCALE + ZOOM_EPSILON) return 'grab';
  if (maxScale > FIT_TRANSFORM_SCALE + ZOOM_EPSILON) return 'zoom-in';
  return 'default';
};

/**
 * Reusable image inspection surface with output-pixel zoom, pan gestures,
 * keyboard controls and a high-detail screen-space renderer. Consumers own
 * only their overlays and optional trailing toolbar actions.
 */
const ZoomableImagePreview = forwardRef<ZoomableImagePreviewHandle, ZoomableImagePreviewProps>(
  function ZoomableImagePreview(
    {
      alt,
      ariaDescribedBy,
      children,
      className,
      detailSrc = null,
      interactionSrc,
      labels,
      sourceSize = null,
      toolbarEnd,
    },
    forwardedRef,
  ) {
    const viewportRef = useRef<HTMLDivElement | null>(null);
    const stageRef = useRef<HTMLDivElement | null>(null);
    const imageRef = useRef<HTMLImageElement | null>(null);
    const screenPreviewRef = useRef<HTMLDivElement | null>(null);
    const zoomLabelRef = useRef<HTMLSpanElement | null>(null);
    const zoomInButtonRef = useRef<HTMLButtonElement | null>(null);
    const zoomOutButtonRef = useRef<HTMLButtonElement | null>(null);
    const transformRef = useRef<PreviewTransform>({ ...IDENTITY_TRANSFORM });
    const geometryRef = useRef<PreviewGeometry>({
      imageHeight: 0,
      imageWidth: 0,
      viewportHeight: 0,
      viewportWidth: 0,
    });
    const fitPixelZoomRef = useRef(1);
    const maxTransformScaleRef = useRef(MAX_PIXEL_ZOOM);
    const renderFrameRef = useRef<number | null>(null);
    const settleTimerRef = useRef<number | null>(null);
    const transitionTimerRef = useRef<number | null>(null);
    const screenPreviewReadyFrameRef = useRef<number | null>(null);
    const shouldAnimateNextFrameRef = useRef(false);
    const activePointerRef = useRef<number | null>(null);
    const lastPointerPositionRef = useRef({ x: 0, y: 0 });
    const detailNaturalSizeRef = useRef({ height: 0, width: 0 });

    const resolveScreenPreviewGeometry = useCallback((transform: PreviewTransform) => {
      const geometry = geometryRef.current;
      if (
        geometry.imageWidth <= 0 ||
        geometry.imageHeight <= 0 ||
        geometry.viewportWidth <= 0 ||
        geometry.viewportHeight <= 0
      ) {
        return null;
      }

      const viewportCenterX = geometry.viewportWidth / 2;
      const viewportCenterY = geometry.viewportHeight / 2;
      return calculateScreenSpacePreviewGeometry({
        imageHeight: geometry.imageHeight,
        imageOffsetX: (geometry.viewportWidth - geometry.imageWidth) / 2,
        imageOffsetY: (geometry.viewportHeight - geometry.imageHeight) / 2,
        imageWidth: geometry.imageWidth,
        positionX: transform.x + viewportCenterX * (1 - transform.scale),
        positionY: transform.y + viewportCenterY * (1 - transform.scale),
        transformScale: transform.scale,
      });
    }, []);

    const {
      bakePreview: bakeScreenPreview,
      resetBakedPreview: resetBakedScreenPreview,
      transformPreview: transformScreenPreview,
    } = useScreenSpacePreviewTransform({
      previewRef: screenPreviewRef,
      resolveGeometry: resolveScreenPreviewGeometry,
    });

    const updateControls = useCallback((transform: PreviewTransform) => {
      const pixelZoom = calculatePixelZoom(transform.scale, fitPixelZoomRef.current);
      if (zoomLabelRef.current) zoomLabelRef.current.textContent = `${Math.round(pixelZoom * 100)}%`;
      if (zoomOutButtonRef.current) {
        zoomOutButtonRef.current.disabled = transform.scale <= FIT_TRANSFORM_SCALE + ZOOM_EPSILON;
      }
      if (zoomInButtonRef.current) {
        zoomInButtonRef.current.disabled = transform.scale >= maxTransformScaleRef.current - ZOOM_EPSILON;
      }
      if (viewportRef.current) {
        viewportRef.current.style.cursor = resolvePreviewCursor(
          activePointerRef.current,
          transform.scale,
          maxTransformScaleRef.current,
        );
      }
    }, []);

    const paintTransform = useCallback(
      (animate: boolean) => {
        shouldAnimateNextFrameRef.current = animate;
        if (renderFrameRef.current !== null) return;

        renderFrameRef.current = requestAnimationFrame(() => {
          renderFrameRef.current = null;
          const stage = stageRef.current;
          const screenPreview = screenPreviewRef.current;
          const transform = transformRef.current;
          if (!stage) return;

          if (transitionTimerRef.current !== null) {
            window.clearTimeout(transitionTimerRef.current);
            transitionTimerRef.current = null;
          }
          if (shouldAnimateNextFrameRef.current) {
            const transition = `transform ${TRANSFORM_ANIMATION_MS}ms cubic-bezier(0.22, 1, 0.36, 1)`;
            stage.style.transition = transition;
            if (screenPreview) screenPreview.style.transition = transition;
            transitionTimerRef.current = window.setTimeout(() => {
              transitionTimerRef.current = null;
              if (stageRef.current) stageRef.current.style.transition = 'none';
              if (screenPreviewRef.current) screenPreviewRef.current.style.transition = 'none';
            }, TRANSFORM_ANIMATION_MS);
          } else {
            stage.style.transition = 'none';
            if (screenPreview) screenPreview.style.transition = 'none';
          }

          stage.style.transform = `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${transform.scale})`;
          transformScreenPreview(transform);
          updateControls(transform);
        });
      },
      [transformScreenPreview, updateControls],
    );

    const beginInteraction = useCallback(() => {
      const viewport = viewportRef.current;
      if (!viewport || viewport.dataset.interacting === 'true') return;
      viewport.dataset.interacting = 'true';
    }, []);

    const settleTransform = useCallback(() => {
      const viewport = viewportRef.current;
      if (!viewport) return;
      if (screenPreviewRef.current) screenPreviewRef.current.style.transition = 'none';
      bakeScreenPreview(transformRef.current);
      viewport.dataset.interacting = 'false';
    }, [bakeScreenPreview]);

    const scheduleSettle = useCallback(() => {
      if (settleTimerRef.current !== null) window.clearTimeout(settleTimerRef.current);
      settleTimerRef.current = window.setTimeout(() => {
        settleTimerRef.current = null;
        settleTransform();
      }, TRANSFORM_SETTLE_DELAY_MS);
    }, [settleTransform]);

    const clampTransform = useCallback((candidate: PreviewTransform): PreviewTransform => {
      const geometry = geometryRef.current;
      const scale = clampTransformScale(candidate.scale, maxTransformScaleRef.current);
      if (
        geometry.imageWidth <= 0 ||
        geometry.imageHeight <= 0 ||
        geometry.viewportWidth <= 0 ||
        geometry.viewportHeight <= 0
      ) {
        return scale <= FIT_TRANSFORM_SCALE ? { ...IDENTITY_TRANSFORM } : { ...candidate, scale };
      }

      const maxX = Math.max(0, (geometry.imageWidth * scale - geometry.viewportWidth) / 2);
      const maxY = Math.max(0, (geometry.imageHeight * scale - geometry.viewportHeight) / 2);
      return {
        scale,
        x: Math.min(maxX, Math.max(-maxX, candidate.x)),
        y: Math.min(maxY, Math.max(-maxY, candidate.y)),
      };
    }, []);

    const applyTransform = useCallback(
      (candidate: PreviewTransform, animate = false) => {
        beginInteraction();
        transformRef.current = clampTransform(candidate);
        paintTransform(animate);
        scheduleSettle();
      },
      [beginInteraction, clampTransform, paintTransform, scheduleSettle],
    );

    const updateGeometry = useCallback(() => {
      const viewport = viewportRef.current;
      const image = imageRef.current;
      if (!viewport || !image || !image.complete || image.naturalWidth === 0) return;

      geometryRef.current = {
        imageHeight: image.clientHeight,
        imageWidth: image.clientWidth,
        viewportHeight: viewport.clientHeight,
        viewportWidth: viewport.clientWidth,
      };
      if (image.clientWidth > 0 && image.clientHeight > 0) {
        const sourceWidth = sourceSize?.width ?? (detailNaturalSizeRef.current.width || image.naturalWidth);
        const sourceHeight = sourceSize?.height ?? (detailNaturalSizeRef.current.height || image.naturalHeight);
        fitPixelZoomRef.current = calculateFitPixelZoom({
          devicePixelRatio: window.devicePixelRatio || 1,
          displayHeight: image.clientHeight,
          displayWidth: image.clientWidth,
          sourceHeight,
          sourceWidth,
        });
        maxTransformScaleRef.current = calculateMaxTransformScale(fitPixelZoomRef.current);
      }
      applyTransform(transformRef.current);
    }, [applyTransform, sourceSize?.height, sourceSize?.width]);

    const handleScreenPreviewReady = useCallback(
      (_isViewportPatch: boolean, image?: HTMLImageElement) => {
        if (image && image.naturalWidth > 0 && image.naturalHeight > 0) {
          detailNaturalSizeRef.current = { height: image.naturalHeight, width: image.naturalWidth };
        }
        updateGeometry();

        if (screenPreviewReadyFrameRef.current !== null) {
          cancelAnimationFrame(screenPreviewReadyFrameRef.current);
        }
        screenPreviewReadyFrameRef.current = requestAnimationFrame(() => {
          screenPreviewReadyFrameRef.current = requestAnimationFrame(() => {
            screenPreviewReadyFrameRef.current = null;
            bakeScreenPreview(transformRef.current);
            if (viewportRef.current) viewportRef.current.dataset.detailReady = 'true';
          });
        });
      },
      [bakeScreenPreview, updateGeometry],
    );

    const resetView = useCallback(() => applyTransform({ ...IDENTITY_TRANSFORM }, true), [applyTransform]);

    const updateZoomFromCenter = useCallback(
      (nextScale: number, animate = true) => {
        const current = transformRef.current;
        const scale = clampTransformScale(nextScale, maxTransformScaleRef.current);
        const ratio = scale / current.scale;
        applyTransform({ scale, x: current.x * ratio, y: current.y * ratio }, animate);
      },
      [applyTransform],
    );

    const stepZoom = useCallback(
      (direction: -1 | 1) => {
        updateZoomFromCenter(
          resolveZoomStep(transformRef.current.scale, direction, fitPixelZoomRef.current, maxTransformScaleRef.current),
        );
      },
      [updateZoomFromCenter],
    );

    const zoomToOneHundredPercent = useCallback(() => {
      updateZoomFromCenter(resolvePixelZoomScale(1, fitPixelZoomRef.current, maxTransformScaleRef.current));
    }, [updateZoomFromCenter]);

    useImperativeHandle(
      forwardedRef,
      () => ({
        fitToViewport: resetView,
        zoomIn: () => stepZoom(1),
        zoomOut: () => stepZoom(-1),
        zoomToActualSize: zoomToOneHundredPercent,
      }),
      [resetView, stepZoom, zoomToOneHundredPercent],
    );

    useEffect(() => {
      transformRef.current = { ...IDENTITY_TRANSFORM };
      fitPixelZoomRef.current = 1;
      maxTransformScaleRef.current = MAX_PIXEL_ZOOM;
      detailNaturalSizeRef.current = { height: 0, width: 0 };
      resetBakedScreenPreview();
      if (viewportRef.current) {
        viewportRef.current.dataset.detailReady = 'false';
      }
      paintTransform(false);
    }, [detailSrc, interactionSrc, paintTransform, resetBakedScreenPreview]);

    useEffect(() => {
      const viewport = viewportRef.current;
      const image = imageRef.current;
      if (!viewport || !image) return;

      const observer = new ResizeObserver(updateGeometry);
      observer.observe(viewport);
      observer.observe(image);
      updateGeometry();
      return () => observer.disconnect();
    }, [interactionSrc, updateGeometry]);

    useEffect(() => {
      const viewport = viewportRef.current;
      if (!viewport) return;

      const handleWheel = (event: WheelEvent) => {
        event.preventDefault();
        event.stopPropagation();
        const current = transformRef.current;
        const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
        const exponent = Math.max(-0.5, Math.min(0.5, -delta * 0.002));
        const scale = clampTransformScale(current.scale * Math.exp(exponent), maxTransformScaleRef.current);
        if (Math.abs(scale - current.scale) < 0.0001) return;

        const bounds = viewport.getBoundingClientRect();
        const pointerX = event.clientX - bounds.left - bounds.width / 2;
        const pointerY = event.clientY - bounds.top - bounds.height / 2;
        const ratio = scale / current.scale;
        applyTransform(
          {
            scale,
            x: pointerX - (pointerX - current.x) * ratio,
            y: pointerY - (pointerY - current.y) * ratio,
          },
          false,
        );
      };

      viewport.addEventListener('wheel', handleWheel, { passive: false });
      return () => viewport.removeEventListener('wheel', handleWheel);
    }, [applyTransform]);

    useEffect(
      () => () => {
        if (renderFrameRef.current !== null) {
          cancelAnimationFrame(renderFrameRef.current);
          renderFrameRef.current = null;
        }
        if (settleTimerRef.current !== null) {
          window.clearTimeout(settleTimerRef.current);
          settleTimerRef.current = null;
        }
        if (transitionTimerRef.current !== null) {
          window.clearTimeout(transitionTimerRef.current);
          transitionTimerRef.current = null;
        }
        if (screenPreviewReadyFrameRef.current !== null) {
          cancelAnimationFrame(screenPreviewReadyFrameRef.current);
          screenPreviewReadyFrameRef.current = null;
        }
      },
      [],
    );

    const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
      if (
        event.button !== 0 ||
        transformRef.current.scale <= FIT_TRANSFORM_SCALE + ZOOM_EPSILON ||
        (event.target instanceof Element && event.target.closest('[data-zoomable-image-preview-toolbar]'))
      ) {
        return;
      }
      event.preventDefault();
      beginInteraction();
      activePointerRef.current = event.pointerId;
      lastPointerPositionRef.current = { x: event.clientX, y: event.clientY };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.currentTarget.dataset.dragging = 'true';
      updateControls(transformRef.current);
    };

    const handlePointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
      if (activePointerRef.current !== event.pointerId) return;
      event.preventDefault();
      const current = transformRef.current;
      const deltaX = event.clientX - lastPointerPositionRef.current.x;
      const deltaY = event.clientY - lastPointerPositionRef.current.y;
      lastPointerPositionRef.current = { x: event.clientX, y: event.clientY };
      applyTransform({ ...current, x: current.x + deltaX, y: current.y + deltaY });
    };

    const handlePointerEnd = (event: ReactPointerEvent<HTMLDivElement>) => {
      if (activePointerRef.current !== event.pointerId) return;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      activePointerRef.current = null;
      event.currentTarget.dataset.dragging = 'false';
      updateControls(transformRef.current);
      scheduleSettle();
    };

    const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        stepZoom(1);
      } else if (event.key === '-') {
        event.preventDefault();
        stepZoom(-1);
      } else if (event.key === '0') {
        event.preventDefault();
        resetView();
      }
    };

    return (
      <div
        aria-describedby={ariaDescribedBy}
        aria-label={alt}
        className={`zoomable-image-preview relative h-full w-full touch-none overflow-hidden bg-[#101010] select-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-inset focus-visible:outline-accent ${className ?? ''}`}
        data-dragging="false"
        data-detail-ready="false"
        data-interacting="false"
        onDoubleClick={() =>
          transformRef.current.scale > FIT_TRANSFORM_SCALE + ZOOM_EPSILON ? resetView() : zoomToOneHundredPercent()
        }
        onKeyDown={handleKeyDown}
        onPointerCancel={handlePointerEnd}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        ref={viewportRef}
        role="region"
        style={{ cursor: 'zoom-in' }}
        tabIndex={0}
      >
        <div
          className="zoomable-image-preview__stage pointer-events-none absolute inset-0 z-0 flex origin-center items-center justify-center p-4 sm:p-6"
          ref={stageRef}
        >
          <img
            alt={alt}
            className="zoomable-image-preview__image zoomable-image-preview__image--interaction block max-h-full max-w-full object-contain"
            decoding="async"
            draggable={false}
            onLoad={updateGeometry}
            ref={imageRef}
            src={interactionSrc}
          />
        </div>

        <ScreenSpacePreview
          finalPreviewUrl={detailSrc}
          hidden={false}
          imagePath={detailSrc || interactionSrc}
          interactivePatch={null}
          isMaxZoom={false}
          isSliderDragging={false}
          key={detailSrc || interactionSrc}
          onProcessedFrameReady={handleScreenPreviewReady}
          ref={screenPreviewRef}
          showOriginal={false}
          thumbnailUrl={interactionSrc}
          transformedOriginalUrl={null}
        />

        {children}

        <div
          aria-label={labels.toolbar ?? alt}
          className="absolute right-3 bottom-3 z-20 flex h-10 items-center overflow-hidden rounded-lg bg-black/85 text-white shadow-lg"
          data-zoomable-image-preview-toolbar
          onDoubleClick={(event) => event.stopPropagation()}
          role="toolbar"
        >
          <button
            aria-label={labels.zoomOut}
            className="flex h-10 w-10 items-center justify-center text-white/70 transition-colors hover:bg-white/10 hover:text-white active:bg-white/15 disabled:opacity-35"
            data-tooltip={labels.zoomOut}
            disabled={transformRef.current.scale <= FIT_TRANSFORM_SCALE + ZOOM_EPSILON}
            onClick={() => stepZoom(-1)}
            ref={zoomOutButtonRef}
            type="button"
          >
            <ZoomOut aria-hidden="true" size={17} />
          </button>
          <span
            className="flex h-10 min-w-14 items-center justify-center border-x border-white/10 px-2 text-center text-[11px] font-medium tabular-nums text-white/85"
            ref={zoomLabelRef}
          >
            {Math.round(calculatePixelZoom(transformRef.current.scale, fitPixelZoomRef.current) * 100)}%
          </span>
          <button
            aria-label={labels.zoomIn}
            className="flex h-10 w-10 items-center justify-center text-white/70 transition-colors hover:bg-white/10 hover:text-white active:bg-white/15 disabled:opacity-35"
            data-tooltip={labels.zoomIn}
            disabled={transformRef.current.scale >= maxTransformScaleRef.current - ZOOM_EPSILON}
            onClick={() => stepZoom(1)}
            ref={zoomInButtonRef}
            type="button"
          >
            <ZoomIn aria-hidden="true" size={17} />
          </button>
          <button
            aria-label={labels.fit}
            className="flex h-10 w-10 items-center justify-center text-white/70 transition-colors hover:bg-white/10 hover:text-white"
            data-tooltip={labels.fit}
            onClick={resetView}
            type="button"
          >
            <ScanSearch aria-hidden="true" size={16} />
          </button>
          {toolbarEnd}
        </div>
      </div>
    );
  },
);

ZoomableImagePreview.displayName = 'ZoomableImagePreview';

export default memo(ZoomableImagePreview);
