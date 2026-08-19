import { memo, useCallback, useEffect, useRef } from 'react';
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import { Maximize2, Minimize2, ScanSearch, ZoomIn, ZoomOut } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface ImageStackResultPreviewProps {
  alignmentLabel: string;
  alt: string;
  detailSrc: string | null;
  isFocused: boolean;
  modeLabel: string;
  onFocusedChange(isFocused: boolean): void;
  src: string;
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

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const ZOOM_STEP = 0.25;
const TRANSFORM_SETTLE_DELAY_MS = 200;
const TRANSFORM_ANIMATION_MS = 160;
const IDENTITY_TRANSFORM: PreviewTransform = { scale: MIN_ZOOM, x: 0, y: 0 };

const clampZoom = (zoom: number, maxZoom: number) => Math.min(maxZoom, Math.max(MIN_ZOOM, zoom));

function ImageStackResultPreview({
  alignmentLabel,
  alt,
  detailSrc,
  isFocused,
  modeLabel,
  onFocusedChange,
  src,
}: ImageStackResultPreviewProps) {
  const { t } = useTranslation();
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const detailImageRef = useRef<HTMLImageElement | null>(null);
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
  const availableMaxZoomRef = useRef(MAX_ZOOM);
  const renderFrameRef = useRef<number | null>(null);
  const settleTimerRef = useRef<number | null>(null);
  const transitionTimerRef = useRef<number | null>(null);
  const shouldAnimateNextFrameRef = useRef(false);
  const activePointerRef = useRef<number | null>(null);
  const lastPointerPositionRef = useRef({ x: 0, y: 0 });

  const updateControls = useCallback((transform: PreviewTransform) => {
    if (zoomLabelRef.current) zoomLabelRef.current.textContent = `${Math.round(transform.scale * 100)}%`;
    if (zoomOutButtonRef.current) zoomOutButtonRef.current.disabled = transform.scale <= MIN_ZOOM + 0.001;
    if (zoomInButtonRef.current) {
      zoomInButtonRef.current.disabled = transform.scale >= availableMaxZoomRef.current - 0.001;
    }
    if (viewportRef.current) {
      viewportRef.current.style.cursor =
        activePointerRef.current !== null ? 'grabbing' : transform.scale > MIN_ZOOM + 0.001 ? 'grab' : 'zoom-in';
    }
  }, []);

  const paintTransform = useCallback(
    (animate: boolean) => {
      shouldAnimateNextFrameRef.current = animate;
      if (renderFrameRef.current !== null) return;

      renderFrameRef.current = requestAnimationFrame(() => {
        renderFrameRef.current = null;
        const stage = stageRef.current;
        const transform = transformRef.current;
        if (!stage) return;

        if (transitionTimerRef.current !== null) {
          window.clearTimeout(transitionTimerRef.current);
          transitionTimerRef.current = null;
        }
        if (shouldAnimateNextFrameRef.current) {
          stage.style.transition = `transform ${TRANSFORM_ANIMATION_MS}ms cubic-bezier(0.22, 1, 0.36, 1)`;
          transitionTimerRef.current = window.setTimeout(() => {
            transitionTimerRef.current = null;
            if (stageRef.current) stageRef.current.style.transition = 'none';
          }, TRANSFORM_ANIMATION_MS);
        } else {
          stage.style.transition = 'none';
        }

        stage.style.transform = `translate3d(${transform.x}px, ${transform.y}px, 0) scale(${transform.scale})`;
        updateControls(transform);
      });
    },
    [updateControls],
  );

  const scheduleSettle = useCallback(() => {
    const viewport = viewportRef.current;
    if (viewport) viewport.dataset.interacting = 'true';
    if (settleTimerRef.current !== null) window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = window.setTimeout(() => {
      settleTimerRef.current = null;
      if (viewportRef.current) viewportRef.current.dataset.interacting = 'false';
    }, TRANSFORM_SETTLE_DELAY_MS);
  }, []);

  const clampTransform = useCallback((candidate: PreviewTransform): PreviewTransform => {
    const geometry = geometryRef.current;
    const scale = clampZoom(candidate.scale, availableMaxZoomRef.current);
    if (
      geometry.imageWidth <= 0 ||
      geometry.imageHeight <= 0 ||
      geometry.viewportWidth <= 0 ||
      geometry.viewportHeight <= 0
    ) {
      return scale <= MIN_ZOOM ? { ...IDENTITY_TRANSFORM } : { ...candidate, scale };
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
      transformRef.current = clampTransform(candidate);
      paintTransform(animate);
      scheduleSettle();
    },
    [clampTransform, paintTransform, scheduleSettle],
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
    const detailImage = detailImageRef.current;
    if (detailImage) {
      detailImage.style.width = `${image.clientWidth}px`;
      detailImage.style.height = `${image.clientHeight}px`;
    }
    if (image.clientWidth > 0 && image.clientHeight > 0) {
      const resolutionImage = detailImage?.complete && detailImage.naturalWidth > 0 ? detailImage : image;
      const nativeWidthZoom = resolutionImage.naturalWidth / image.clientWidth;
      const nativeHeightZoom = resolutionImage.naturalHeight / image.clientHeight;
      availableMaxZoomRef.current = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nativeWidthZoom, nativeHeightZoom));
    }
    applyTransform(transformRef.current);
  }, [applyTransform]);

  const resetView = useCallback(() => applyTransform({ ...IDENTITY_TRANSFORM }, true), [applyTransform]);

  const updateZoomFromCenter = useCallback(
    (nextZoom: number, animate = true) => {
      const current = transformRef.current;
      const scale = clampZoom(nextZoom, availableMaxZoomRef.current);
      const ratio = scale / current.scale;
      applyTransform({ scale, x: current.x * ratio, y: current.y * ratio }, animate);
    },
    [applyTransform],
  );

  useEffect(() => {
    transformRef.current = { ...IDENTITY_TRANSFORM };
    availableMaxZoomRef.current = MAX_ZOOM;
    if (viewportRef.current) viewportRef.current.dataset.detailReady = 'false';
    paintTransform(false);
  }, [detailSrc, paintTransform, src]);

  useEffect(() => {
    const viewport = viewportRef.current;
    const image = imageRef.current;
    if (!viewport || !image) return;

    const observer = new ResizeObserver(updateGeometry);
    observer.observe(viewport);
    observer.observe(image);
    updateGeometry();
    return () => observer.disconnect();
  }, [src, updateGeometry]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;

    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      event.stopPropagation();
      const current = transformRef.current;
      const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
      const exponent = Math.max(-0.5, Math.min(0.5, -delta * 0.002));
      const scale = clampZoom(current.scale * Math.exp(exponent), availableMaxZoomRef.current);
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
      if (renderFrameRef.current !== null) cancelAnimationFrame(renderFrameRef.current);
      if (settleTimerRef.current !== null) window.clearTimeout(settleTimerRef.current);
      if (transitionTimerRef.current !== null) window.clearTimeout(transitionTimerRef.current);
    },
    [],
  );

  const handlePointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || transformRef.current.scale <= MIN_ZOOM + 0.001) return;
    event.preventDefault();
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
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      updateZoomFromCenter(transformRef.current.scale + ZOOM_STEP);
    } else if (event.key === '-') {
      event.preventDefault();
      updateZoomFromCenter(transformRef.current.scale - ZOOM_STEP);
    } else if (event.key === '0') {
      event.preventDefault();
      resetView();
    }
  };

  return (
    <div
      aria-describedby="image-stack-preview-hint"
      aria-label={alt}
      className="image-stack-preview-surface relative h-full w-full touch-none overflow-hidden bg-[#101010] select-none focus-visible:outline focus-visible:outline-2 focus-visible:outline-inset focus-visible:outline-accent"
      data-dragging="false"
      data-detail-ready="false"
      data-interacting="false"
      onDoubleClick={() =>
        transformRef.current.scale > MIN_ZOOM + 0.001
          ? resetView()
          : updateZoomFromCenter(Math.min(2, availableMaxZoomRef.current))
      }
      onKeyDown={handleKeyDown}
      onPointerCancel={handlePointerEnd}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerEnd}
      ref={viewportRef}
      role="region"
      style={{ cursor: transformRef.current.scale > MIN_ZOOM + 0.001 ? 'grab' : 'zoom-in' }}
      tabIndex={0}
    >
      <div
        className="image-stack-preview-stage pointer-events-none absolute inset-0 flex origin-center items-center justify-center p-4 sm:p-6"
        ref={stageRef}
      >
        <img
          alt={alt}
          className="image-stack-preview-image image-stack-preview-image--interaction block max-h-full max-w-full object-contain"
          decoding="async"
          draggable={false}
          onLoad={updateGeometry}
          ref={imageRef}
          src={src}
        />
        {detailSrc && detailSrc !== src && (
          <img
            alt=""
            aria-hidden="true"
            className="image-stack-preview-image image-stack-preview-image--detail absolute top-1/2 left-1/2 object-fill"
            decoding="async"
            draggable={false}
            onError={() => {
              if (viewportRef.current) viewportRef.current.dataset.detailReady = 'false';
            }}
            onLoad={() => {
              if (viewportRef.current) viewportRef.current.dataset.detailReady = 'true';
              updateGeometry();
            }}
            ref={detailImageRef}
            src={detailSrc}
            style={{ transform: 'translate3d(-50%, -50%, 0)' }}
          />
        )}
      </div>

      <div className="pointer-events-none absolute top-3 left-3 z-10 flex flex-wrap gap-1.5">
        <span className="rounded-md border border-white/10 bg-black/80 px-2 py-1 text-[11px] font-medium text-white/90">
          {modeLabel}
        </span>
        <span className="rounded-md border border-white/10 bg-black/80 px-2 py-1 text-[11px] text-white/70">
          {alignmentLabel}
        </span>
      </div>

      <p
        className="pointer-events-none absolute bottom-3 left-3 z-10 hidden rounded-md bg-black/75 px-2 py-1 text-[10px] text-white/65 sm:block"
        id="image-stack-preview-hint"
      >
        {t('modals.imageStack.previewHint')}
      </p>

      <div
        aria-label={alt}
        className="absolute right-3 bottom-3 z-20 flex h-10 items-center overflow-hidden rounded-lg border border-white/15 bg-black/85 text-white shadow-lg"
        onDoubleClick={(event) => event.stopPropagation()}
        onPointerDown={(event) => event.stopPropagation()}
        role="toolbar"
      >
        <button
          aria-label={t('modals.imageStack.zoomOut')}
          className="flex h-10 w-10 items-center justify-center text-white/70 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-35"
          data-tooltip={t('modals.imageStack.zoomOut')}
          disabled={transformRef.current.scale <= MIN_ZOOM + 0.001}
          onClick={() => updateZoomFromCenter(transformRef.current.scale - ZOOM_STEP)}
          ref={zoomOutButtonRef}
          type="button"
        >
          <ZoomOut aria-hidden="true" size={17} />
        </button>
        <span
          className="flex h-10 min-w-14 items-center justify-center border-x border-white/10 px-2 text-center text-[11px] font-medium tabular-nums text-white/85"
          ref={zoomLabelRef}
        >
          {Math.round(transformRef.current.scale * 100)}%
        </span>
        <button
          aria-label={t('modals.imageStack.zoomIn')}
          className="flex h-10 w-10 items-center justify-center text-white/70 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-35"
          data-tooltip={t('modals.imageStack.zoomIn')}
          disabled={transformRef.current.scale >= availableMaxZoomRef.current - 0.001}
          onClick={() => updateZoomFromCenter(transformRef.current.scale + ZOOM_STEP)}
          ref={zoomInButtonRef}
          type="button"
        >
          <ZoomIn aria-hidden="true" size={17} />
        </button>
        <button
          aria-label={t('modals.imageStack.fitPreview')}
          className="flex h-10 w-10 items-center justify-center border-l border-white/10 text-white/70 transition-colors hover:bg-white/10 hover:text-white"
          data-tooltip={t('modals.imageStack.fitPreview')}
          onClick={resetView}
          type="button"
        >
          <ScanSearch aria-hidden="true" size={16} />
        </button>
        <button
          aria-label={isFocused ? t('modals.imageStack.exitFocusPreview') : t('modals.imageStack.focusPreview')}
          className={`flex h-10 w-10 items-center justify-center border-l border-white/10 transition-colors ${
            isFocused ? 'bg-white/15 text-white' : 'text-white/70 hover:bg-white/10 hover:text-white'
          }`}
          data-tooltip={isFocused ? t('modals.imageStack.exitFocusPreview') : t('modals.imageStack.focusPreview')}
          onClick={() => onFocusedChange(!isFocused)}
          type="button"
        >
          {isFocused ? <Minimize2 aria-hidden="true" size={17} /> : <Maximize2 aria-hidden="true" size={17} />}
        </button>
      </div>
    </div>
  );
}

export default memo(ImageStackResultPreview);
