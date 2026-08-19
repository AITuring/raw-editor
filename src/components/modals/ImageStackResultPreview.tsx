import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  KeyboardEvent as ReactKeyboardEvent,
  MouseEvent as ReactMouseEvent,
  WheelEvent as ReactWheelEvent,
} from 'react';
import { Maximize2, Minimize2, ScanSearch, ZoomIn, ZoomOut } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface ImageStackResultPreviewProps {
  alignmentLabel: string;
  alt: string;
  isFocused: boolean;
  modeLabel: string;
  onFocusedChange(isFocused: boolean): void;
  src: string;
}

interface PreviewView {
  fitScale: number;
  maxZoom: number;
  panX: number;
  panY: number;
  zoom: number;
}

const MIN_ZOOM = 1;
const MAX_RENDER_SCALE = 4;
const ZOOM_FACTOR = 1.25;
const PREVIEW_PADDING = 24;
const MAX_CANVAS_DPR = 2;
const SETTLE_DELAY_MS = 90;

const clampZoom = (zoom: number, maxZoom: number) => Math.min(maxZoom, Math.max(MIN_ZOOM, zoom));

export default function ImageStackResultPreview({
  alignmentLabel,
  alt,
  isFocused,
  modeLabel,
  onFocusedChange,
  src,
}: ImageStackResultPreviewProps) {
  const { t } = useTranslation();
  const [zoom, setZoom] = useState(MIN_ZOOM);
  const [renderedScale, setRenderedScale] = useState(1);
  const [availableMaxZoom, setAvailableMaxZoom] = useState(MIN_ZOOM);
  const [isDragging, setIsDragging] = useState(false);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sourceImageRef = useRef<HTMLImageElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const settleTimerRef = useRef<number | null>(null);
  const isInteractingRef = useRef(false);
  const lastPointerPosition = useRef({ x: 0, y: 0 });
  const viewRef = useRef<PreviewView>({
    fitScale: 1,
    maxZoom: MIN_ZOOM,
    panX: 0,
    panY: 0,
    zoom: MIN_ZOOM,
  });

  const drawCanvas = useCallback(() => {
    const viewport = viewportRef.current;
    const canvas = canvasRef.current;
    const image = sourceImageRef.current;
    if (!viewport || !canvas) return;

    const viewportWidth = Math.max(1, viewport.clientWidth);
    const viewportHeight = Math.max(1, viewport.clientHeight);
    const dpr = Math.min(MAX_CANVAS_DPR, Math.max(1, window.devicePixelRatio || 1));
    const backingWidth = Math.max(1, Math.round(viewportWidth * dpr));
    const backingHeight = Math.max(1, Math.round(viewportHeight * dpr));
    if (canvas.width !== backingWidth || canvas.height !== backingHeight) {
      canvas.width = backingWidth;
      canvas.height = backingHeight;
    }

    const context = canvas.getContext('2d', { alpha: false });
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.fillStyle = '#101010';
    context.fillRect(0, 0, viewportWidth, viewportHeight);
    if (!image || !image.complete || image.naturalWidth === 0 || image.naturalHeight === 0) return;

    const padding = Math.min(PREVIEW_PADDING, viewportWidth * 0.04, viewportHeight * 0.04);
    const fitScale = Math.max(
      Number.EPSILON,
      Math.min(
        Math.max(1, viewportWidth - padding * 2) / image.naturalWidth,
        Math.max(1, viewportHeight - padding * 2) / image.naturalHeight,
      ),
    );
    const nextMaxZoom = Math.max(MIN_ZOOM, MAX_RENDER_SCALE / fitScale);
    const view = viewRef.current;
    view.fitScale = fitScale;
    view.maxZoom = nextMaxZoom;
    view.zoom = clampZoom(view.zoom, nextMaxZoom);

    const imageScale = fitScale * view.zoom;
    const displayWidth = image.naturalWidth * imageScale;
    const displayHeight = image.naturalHeight * imageScale;
    const maxPanX = Math.max(0, (displayWidth - viewportWidth) / 2);
    const maxPanY = Math.max(0, (displayHeight - viewportHeight) / 2);
    view.panX = Math.min(maxPanX, Math.max(-maxPanX, view.panX));
    view.panY = Math.min(maxPanY, Math.max(-maxPanY, view.panY));

    const destinationX = (viewportWidth - displayWidth) / 2 + view.panX;
    const destinationY = (viewportHeight - displayHeight) / 2 + view.panY;
    const visibleLeft = Math.max(0, destinationX);
    const visibleTop = Math.max(0, destinationY);
    const visibleRight = Math.min(viewportWidth, destinationX + displayWidth);
    const visibleBottom = Math.min(viewportHeight, destinationY + displayHeight);

    if (visibleRight > visibleLeft && visibleBottom > visibleTop) {
      const sourceX = (visibleLeft - destinationX) / imageScale;
      const sourceY = (visibleTop - destinationY) / imageScale;
      const sourceWidth = (visibleRight - visibleLeft) / imageScale;
      const sourceHeight = (visibleBottom - visibleTop) / imageScale;
      context.imageSmoothingEnabled = imageScale < 0.999;
      context.imageSmoothingQuality = isInteractingRef.current ? 'medium' : 'high';
      context.drawImage(
        image,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        visibleLeft,
        visibleTop,
        visibleRight - visibleLeft,
        visibleBottom - visibleTop,
      );
    }

    setAvailableMaxZoom((current) => (Math.abs(current - nextMaxZoom) < 0.001 ? current : nextMaxZoom));
    setRenderedScale((current) => (Math.abs(current - imageScale) < 0.001 ? current : imageScale));
    setZoom((current) => (Math.abs(current - view.zoom) < 0.001 ? current : view.zoom));
  }, []);

  const scheduleRender = useCallback(() => {
    if (animationFrameRef.current !== null) return;
    animationFrameRef.current = window.requestAnimationFrame(() => {
      animationFrameRef.current = null;
      drawCanvas();
    });
  }, [drawCanvas]);

  const markInteraction = useCallback(() => {
    isInteractingRef.current = true;
    if (settleTimerRef.current !== null) window.clearTimeout(settleTimerRef.current);
    settleTimerRef.current = window.setTimeout(() => {
      settleTimerRef.current = null;
      isInteractingRef.current = false;
      scheduleRender();
    }, SETTLE_DELAY_MS);
  }, [scheduleRender]);

  const resetView = useCallback(() => {
    const view = viewRef.current;
    view.zoom = MIN_ZOOM;
    view.panX = 0;
    view.panY = 0;
    setZoom(MIN_ZOOM);
    scheduleRender();
  }, [scheduleRender]);

  const updateZoom = useCallback(
    (nextZoom: number, anchorX = 0, anchorY = 0) => {
      const view = viewRef.current;
      const resolvedZoom = clampZoom(nextZoom, view.maxZoom);
      if (Math.abs(resolvedZoom - view.zoom) < 0.0001) return;
      const scaleRatio = resolvedZoom / view.zoom;
      view.panX = anchorX - (anchorX - view.panX) * scaleRatio;
      view.panY = anchorY - (anchorY - view.panY) * scaleRatio;
      view.zoom = resolvedZoom;
      if (resolvedZoom === MIN_ZOOM) {
        view.panX = 0;
        view.panY = 0;
      }
      markInteraction();
      scheduleRender();
    },
    [markInteraction, scheduleRender],
  );

  useEffect(() => {
    let cancelled = false;
    const image = new Image();
    image.decoding = 'async';
    sourceImageRef.current = null;
    resetView();
    setAvailableMaxZoom(MIN_ZOOM);
    image.onload = () => {
      if (cancelled) return;
      sourceImageRef.current = image;
      resetView();
    };
    image.onerror = () => {
      if (!cancelled) sourceImageRef.current = null;
    };
    image.src = src;
    return () => {
      cancelled = true;
      image.onload = null;
      image.onerror = null;
      if (sourceImageRef.current === image) sourceImageRef.current = null;
    };
  }, [resetView, src]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const observer = new ResizeObserver(scheduleRender);
    observer.observe(viewport);
    scheduleRender();
    return () => observer.disconnect();
  }, [scheduleRender]);

  useEffect(
    () => () => {
      if (animationFrameRef.current !== null) window.cancelAnimationFrame(animationFrameRef.current);
      if (settleTimerRef.current !== null) window.clearTimeout(settleTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (event: MouseEvent) => {
      const deltaX = event.clientX - lastPointerPosition.current.x;
      const deltaY = event.clientY - lastPointerPosition.current.y;
      const view = viewRef.current;
      view.panX += deltaX;
      view.panY += deltaY;
      lastPointerPosition.current = { x: event.clientX, y: event.clientY };
      markInteraction();
      scheduleRender();
    };
    const handleMouseUp = () => {
      setIsDragging(false);
      isInteractingRef.current = false;
      scheduleRender();
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, markInteraction, scheduleRender]);

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const viewport = viewportRef.current;
    if (!viewport) return;

    const normalizedDelta =
      event.deltaMode === 1
        ? event.deltaY * 16
        : event.deltaMode === 2
          ? event.deltaY * viewport.clientHeight
          : event.deltaY;
    const nextZoom = viewRef.current.zoom * Math.exp(-normalizedDelta * 0.0015);
    const bounds = viewport.getBoundingClientRect();
    updateZoom(
      nextZoom,
      event.clientX - bounds.left - bounds.width / 2,
      event.clientY - bounds.top - bounds.height / 2,
    );
  };

  const handleMouseDown = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (event.button !== 0 || viewRef.current.zoom === MIN_ZOOM) return;
    event.preventDefault();
    setIsDragging(true);
    lastPointerPosition.current = { x: event.clientX, y: event.clientY };
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      updateZoom(viewRef.current.zoom * ZOOM_FACTOR);
    } else if (event.key === '-') {
      event.preventDefault();
      updateZoom(viewRef.current.zoom / ZOOM_FACTOR);
    } else if (event.key === '0') {
      event.preventDefault();
      resetView();
    }
  };

  const zoomPercent = Math.round(renderedScale * 100);

  return (
    <div
      aria-describedby="image-stack-preview-hint"
      aria-label={alt}
      className={`relative h-full w-full overflow-hidden select-none bg-[#101010] focus-visible:outline focus-visible:outline-2 focus-visible:outline-inset focus-visible:outline-accent ${
        zoom > MIN_ZOOM ? (isDragging ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-zoom-in'
      }`}
      onDoubleClick={() =>
        viewRef.current.zoom > MIN_ZOOM
          ? resetView()
          : updateZoom(Math.min(viewRef.current.maxZoom, 1 / viewRef.current.fitScale))
      }
      onKeyDown={handleKeyDown}
      onMouseDown={handleMouseDown}
      onWheel={handleWheel}
      ref={viewportRef}
      role="region"
      tabIndex={0}
    >
      <canvas aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-full" ref={canvasRef} />

      <div className="pointer-events-none absolute left-3 top-3 z-10 flex flex-wrap gap-1.5">
        <span className="rounded-md border border-white/10 bg-black/65 px-2 py-1 text-[11px] font-medium text-white/90 backdrop-blur-sm">
          {modeLabel}
        </span>
        <span className="rounded-md border border-white/10 bg-black/65 px-2 py-1 text-[11px] text-white/70 backdrop-blur-sm">
          {alignmentLabel}
        </span>
      </div>

      <p
        className="pointer-events-none absolute bottom-3 left-3 z-10 hidden rounded-md bg-black/55 px-2 py-1 text-[10px] text-white/65 backdrop-blur-sm sm:block"
        id="image-stack-preview-hint"
      >
        {t('modals.imageStack.previewHint')}
      </p>

      <div
        aria-label={alt}
        className="absolute bottom-3 right-3 z-20 flex h-10 items-center overflow-hidden rounded-lg border border-white/15 bg-black/70 text-white shadow-lg backdrop-blur-md"
        onDoubleClick={(event) => event.stopPropagation()}
        onMouseDown={(event) => event.stopPropagation()}
        role="toolbar"
      >
        <button
          aria-label={t('modals.imageStack.zoomOut')}
          className="flex h-10 w-10 items-center justify-center text-white/70 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-35"
          data-tooltip={t('modals.imageStack.zoomOut')}
          disabled={zoom <= MIN_ZOOM}
          onClick={() => updateZoom(viewRef.current.zoom / ZOOM_FACTOR)}
          type="button"
        >
          <ZoomOut aria-hidden="true" size={17} />
        </button>
        <span
          aria-live="polite"
          className="flex h-10 min-w-14 items-center justify-center border-x border-white/10 px-2 text-center text-[11px] font-medium tabular-nums text-white/85"
        >
          {zoomPercent}%
        </span>
        <button
          aria-label={t('modals.imageStack.zoomIn')}
          className="flex h-10 w-10 items-center justify-center text-white/70 transition-colors hover:bg-white/10 hover:text-white disabled:opacity-35"
          data-tooltip={t('modals.imageStack.zoomIn')}
          disabled={zoom >= availableMaxZoom - 0.001}
          onClick={() => updateZoom(viewRef.current.zoom * ZOOM_FACTOR)}
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
