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

const MIN_ZOOM = 1;
const MAX_ZOOM = 8;
const ZOOM_STEP = 0.25;

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
  const [availableMaxZoom, setAvailableMaxZoom] = useState(MAX_ZOOM);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const lastPointerPosition = useRef({ x: 0, y: 0 });

  const resetView = useCallback(() => {
    setZoom(MIN_ZOOM);
    setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    resetView();
    setAvailableMaxZoom(MAX_ZOOM);
  }, [resetView, src]);

  const updateAvailableMaxZoom = useCallback(() => {
    const image = imageRef.current;
    if (!image || !image.complete || image.naturalWidth === 0 || image.clientWidth === 0 || image.clientHeight === 0) {
      return;
    }
    const nativeWidthZoom = image.naturalWidth / image.clientWidth;
    const nativeHeightZoom = image.naturalHeight / image.clientHeight;
    const nextMaxZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, nativeWidthZoom, nativeHeightZoom));
    setAvailableMaxZoom(nextMaxZoom);
    setZoom((current) => Math.min(current, nextMaxZoom));
    if (nextMaxZoom === MIN_ZOOM) setPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    const viewport = viewportRef.current;
    const image = imageRef.current;
    if (!viewport || !image) return;

    const observer = new ResizeObserver(updateAvailableMaxZoom);
    observer.observe(viewport);
    observer.observe(image);
    updateAvailableMaxZoom();
    return () => observer.disconnect();
  }, [src, updateAvailableMaxZoom]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (event: MouseEvent) => {
      const deltaX = event.clientX - lastPointerPosition.current.x;
      const deltaY = event.clientY - lastPointerPosition.current.y;
      setPan((current) => ({ x: current.x + deltaX, y: current.y + deltaY }));
      lastPointerPosition.current = { x: event.clientX, y: event.clientY };
    };
    const handleMouseUp = () => setIsDragging(false);

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  const updateZoomFromCenter = useCallback(
    (nextZoom: number) => {
      const resolvedZoom = clampZoom(nextZoom, availableMaxZoom);
      setZoom(resolvedZoom);
      if (resolvedZoom === MIN_ZOOM) setPan({ x: 0, y: 0 });
    },
    [availableMaxZoom],
  );

  const handleWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!viewportRef.current) return;

    const nextZoom = clampZoom(zoom - event.deltaY * 0.0015, availableMaxZoom);
    if (nextZoom === zoom) return;

    const bounds = viewportRef.current.getBoundingClientRect();
    const pointerX = event.clientX - bounds.left - bounds.width / 2;
    const pointerY = event.clientY - bounds.top - bounds.height / 2;
    const scaleRatio = nextZoom / zoom;

    setPan((current) => ({
      x: pointerX - (pointerX - current.x) * scaleRatio,
      y: pointerY - (pointerY - current.y) * scaleRatio,
    }));
    setZoom(nextZoom);
  };

  const handleMouseDown = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (event.button !== 0 || zoom === MIN_ZOOM) return;
    event.preventDefault();
    setIsDragging(true);
    lastPointerPosition.current = { x: event.clientX, y: event.clientY };
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      updateZoomFromCenter(zoom + ZOOM_STEP);
    } else if (event.key === '-') {
      event.preventDefault();
      updateZoomFromCenter(zoom - ZOOM_STEP);
    } else if (event.key === '0') {
      event.preventDefault();
      resetView();
    }
  };

  const zoomPercent = Math.round(zoom * 100);

  return (
    <div
      aria-describedby="image-stack-preview-hint"
      aria-label={alt}
      className={`relative h-full w-full overflow-hidden select-none bg-[#101010] focus-visible:outline focus-visible:outline-2 focus-visible:outline-inset focus-visible:outline-accent ${
        zoom > MIN_ZOOM ? (isDragging ? 'cursor-grabbing' : 'cursor-grab') : 'cursor-zoom-in'
      }`}
      onDoubleClick={() => (zoom > MIN_ZOOM ? resetView() : updateZoomFromCenter(2))}
      onKeyDown={handleKeyDown}
      onMouseDown={handleMouseDown}
      onWheel={handleWheel}
      ref={viewportRef}
      role="region"
      tabIndex={0}
    >
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-4 sm:p-6">
        <div
          className="flex h-full w-full origin-center items-center justify-center"
          style={{
            transform: `translate3d(${pan.x}px, ${pan.y}px, 0) scale(${zoom})`,
            transition: isDragging ? 'none' : 'transform 100ms cubic-bezier(0.22, 1, 0.36, 1)',
          }}
        >
          <img
            alt={alt}
            className="block max-h-full max-w-full object-contain shadow-2xl"
            draggable={false}
            onLoad={updateAvailableMaxZoom}
            ref={imageRef}
            src={src}
          />
        </div>
      </div>

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
          onClick={() => updateZoomFromCenter(zoom - ZOOM_STEP)}
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
          onClick={() => updateZoomFromCenter(zoom + ZOOM_STEP)}
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
