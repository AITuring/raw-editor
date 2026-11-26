import { memo, useEffect, useId, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { Maximize2, Minimize2, SquareDashedMousePointer } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ImageStackResultSize, ImageStackSoftRegions } from '../../store/useUIStore';
import type { StackReportInspectionMarker } from '../../utils/stackReportInspection';
import type {
  CoordinateFrame,
  ManualReviewRegion,
  NormalizedBox,
  SourceLock,
  WorldBox,
} from '../../utils/sourceLockSidecar';
import { normalizedBoxToWorldBox, worldBoxToPolygon, worldPolygonToNormalizedBox } from '../../utils/sourceLockSidecar';
import { ZoomableImagePreview } from '../preview';
import type { ZoomableImagePreviewHandle } from '../preview/ZoomableImagePreview';

interface ImageStackResultPreviewProps {
  alignmentLabel: string;
  alt: string;
  coordinateFrame?: CoordinateFrame | null;
  detailSrc: string | null;
  draftWorldBox?: WorldBox | null;
  focusedMarker?: StackReportInspectionMarker | null;
  focusedWorldBox?: WorldBox | null;
  inspectionMarkers?: StackReportInspectionMarker[] | null;
  isDrawingRegion?: boolean;
  isFocused: boolean;
  manualReviewRegions?: ManualReviewRegion[] | null;
  modeLabel: string;
  onDrawnWorldBox?(worldBox: WorldBox): void;
  onFocusedChange(isFocused: boolean): void;
  onSelectMarker?(marker: StackReportInspectionMarker): void;
  onToggleDrawRegion?(): void;
  resultSize: ImageStackResultSize | null;
  softRegions?: ImageStackSoftRegions | null;
  sourceLocks?: SourceLock[] | null;
  skippedSources?: string[] | null;
  src: string;
  stationsUnrelated?: boolean;
}

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

function ImageStackResultPreview({
  alignmentLabel,
  alt,
  coordinateFrame = null,
  detailSrc,
  draftWorldBox = null,
  focusedMarker = null,
  focusedWorldBox = null,
  inspectionMarkers = null,
  isDrawingRegion = false,
  isFocused,
  manualReviewRegions = null,
  modeLabel,
  onDrawnWorldBox,
  onFocusedChange,
  onSelectMarker,
  onToggleDrawRegion,
  resultSize,
  softRegions = null,
  sourceLocks = null,
  skippedSources = null,
  src,
  stationsUnrelated = false,
}: ImageStackResultPreviewProps) {
  const { t } = useTranslation();
  const hintId = useId();
  // The soft regions are circled over the result itself; a circle zooms the
  // preview onto its region.
  const previewRef = useRef<ZoomableImagePreviewHandle | null>(null);
  const [showSoftRegions, setShowSoftRegions] = useState(true);
  const [showLockOverlays, setShowLockOverlays] = useState(true);
  const [dragOrigin, setDragOrigin] = useState<{ x: number; y: number } | null>(null);
  const [dragRect, setDragRect] = useState<NormalizedBox | null>(null);
  useEffect(() => setShowSoftRegions(true), [src]);
  useEffect(() => {
    if (!isDrawingRegion) {
      setDragOrigin(null);
      setDragRect(null);
    }
  }, [isDrawingRegion]);
  useEffect(() => {
    if (!focusedMarker) return;
    const { x, y, width, height } = focusedMarker.normalizedBox;
    previewRef.current?.zoomToRegion(x, y, width, height);
  }, [focusedMarker]);
  useEffect(() => {
    if (!focusedWorldBox || !coordinateFrame) return;
    const normalized = worldPolygonToNormalizedBox(worldBoxToPolygon(focusedWorldBox), coordinateFrame);
    previewRef.current?.zoomToRegion(normalized.x, normalized.y, normalized.width, normalized.height);
  }, [coordinateFrame, focusedWorldBox]);
  const regionFrame =
    softRegions?.outputSize ?? (resultSize ? { width: resultSize.width, height: resultSize.height } : null);
  const showingSoftRegions = showSoftRegions && softRegions !== null && regionFrame !== null;
  const focusLabel = isFocused ? t('modals.imageStack.exitFocusPreview') : t('modals.imageStack.focusPreview');
  const memorySafeScale =
    resultSize?.renderScale && resultSize.renderScale < 0.999 ? Math.round(resultSize.renderScale * 100) : null;
  const fullCanvasLabel =
    resultSize?.fullCanvasWidth && resultSize.fullCanvasHeight
      ? `${resultSize.fullCanvasWidth.toLocaleString()} × ${resultSize.fullCanvasHeight.toLocaleString()}`
      : undefined;
  const softRegionHint = softRegions
    ? t('modals.imageStack.softRegionsHint', {
        sourcesSoft: softRegions.sourcesSoft,
        unaligned: softRegions.count - softRegions.sourcesSoft,
      })
    : undefined;
  const skippedNames =
    skippedSources && skippedSources.length > 0
      ? skippedSources.map((path) => path.split(/[\\/]/).pop() ?? path).join(', ')
      : null;
  const skippedHint = skippedNames ? t('modals.imageStack.skippedSourcesHint', { names: skippedNames }) : undefined;

  const lockRegionCount = (sourceLocks?.length ?? 0) + (manualReviewRegions?.length ?? 0);

  const toNormalizedPoint = (event: ReactPointerEvent<HTMLDivElement>): { x: number; y: number } => {
    const rect = event.currentTarget.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    return {
      x: clamp01((event.clientX - rect.left) / width),
      y: clamp01((event.clientY - rect.top) / height),
    };
  };

  const handleDrawPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!isDrawingRegion || !coordinateFrame) return;
    event.stopPropagation();
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = toNormalizedPoint(event);
    setDragOrigin(start);
    setDragRect({ x: start.x, y: start.y, width: 0, height: 0 });
  };

  const handleDrawPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!isDrawingRegion || !dragOrigin) return;
    event.stopPropagation();
    const current = toNormalizedPoint(event);
    setDragRect({
      x: Math.min(dragOrigin.x, current.x),
      y: Math.min(dragOrigin.y, current.y),
      width: Math.abs(current.x - dragOrigin.x),
      height: Math.abs(current.y - dragOrigin.y),
    });
  };

  const handleDrawPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!isDrawingRegion || !dragOrigin || !coordinateFrame) return;
    event.stopPropagation();
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const end = toNormalizedPoint(event);
    const normalized: NormalizedBox = {
      x: Math.min(dragOrigin.x, end.x),
      y: Math.min(dragOrigin.y, end.y),
      width: Math.max(0.004, Math.abs(end.x - dragOrigin.x)),
      height: Math.max(0.004, Math.abs(end.y - dragOrigin.y)),
    };
    setDragOrigin(null);
    setDragRect(null);
    onDrawnWorldBox?.(normalizedBoxToWorldBox(normalized, coordinateFrame));
  };

  const softRegionOverlay =
    showingSoftRegions && softRegions && regionFrame
      ? softRegions.regions.map((region, index) => {
          const x = region.x / regionFrame.width;
          const y = region.y / regionFrame.height;
          const width = region.width / regionFrame.width;
          const height = region.height / regionFrame.height;
          const label = t(
            region.kind === 'sources_soft'
              ? 'modals.imageStack.softRegionSourcesSoft'
              : 'modals.imageStack.softRegionUnaligned',
            { number: index + 1 },
          );
          // A circle a little larger than its region, never too small to hit.
          return (
            <button
              aria-label={label}
              className={`pointer-events-auto absolute rounded-[50%] ring-2 ring-inset transition-colors ${
                region.kind === 'sources_soft'
                  ? 'ring-red-500/90 hover:bg-red-500/15'
                  : 'ring-orange-400/90 hover:bg-orange-400/15'
              }`}
              data-tooltip={label}
              key={`${region.x}-${region.y}-${index}`}
              onClick={(event) => {
                event.stopPropagation();
                previewRef.current?.zoomToRegion(x, y, width, height);
              }}
              onDoubleClick={(event) => event.stopPropagation()}
              onPointerDown={(event) => event.stopPropagation()}
              style={{
                height: `max(${height * 140}%, 14px)`,
                left: `${(x + width / 2) * 100}%`,
                top: `${(y + height / 2) * 100}%`,
                transform: 'translate(-50%, -50%)',
                width: `max(${width * 140}%, 14px)`,
              }}
              type="button"
            >
              <span
                className={`absolute -top-2 -left-2 flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[9px] font-semibold text-white ${
                  region.kind === 'sources_soft' ? 'bg-red-500' : 'bg-orange-500'
                }`}
              >
                {index + 1}
              </span>
            </button>
          );
        })
      : null;

  const inspectionOverlay =
    inspectionMarkers && inspectionMarkers.length > 0
      ? inspectionMarkers.map((marker) => {
          const { x, y, width, height } = marker.normalizedBox;
          const isSelected = focusedMarker?.id === marker.id;
          return (
            <button
              aria-label={marker.detail}
              className={`pointer-events-auto absolute rounded-sm ring-2 ring-inset transition-colors ${
                marker.kind === 'qg_fail'
                  ? isSelected
                    ? 'bg-red-500/20 ring-red-400'
                    : 'ring-red-500/85 hover:bg-red-500/15'
                  : isSelected
                    ? 'bg-amber-400/20 ring-amber-300'
                    : 'ring-amber-400/85 hover:bg-amber-400/15'
              }`}
              data-tooltip={marker.detail}
              key={marker.id}
              onClick={(event) => {
                event.stopPropagation();
                previewRef.current?.zoomToRegion(x, y, width, height);
                onSelectMarker?.(marker);
              }}
              onDoubleClick={(event) => event.stopPropagation()}
              onPointerDown={(event) => event.stopPropagation()}
              style={{
                height: `max(${height * 100}%, 14px)`,
                left: `${x * 100}%`,
                top: `${y * 100}%`,
                width: `max(${width * 100}%, 14px)`,
              }}
              type="button"
            />
          );
        })
      : null;

  const sourceLocksOverlay =
    showLockOverlays && coordinateFrame && sourceLocks && sourceLocks.length > 0
      ? sourceLocks.map((lock) => {
          const normalized = worldPolygonToNormalizedBox(lock.region.polygon_world, coordinateFrame);
          const isActive = lock.status === 'active';
          const badge = isActive
            ? `${lock.lock_id} · #${lock.requested_sources.join(',')}`
            : `${lock.lock_id} · ${lock.status.toUpperCase()}`;
          return (
            <button
              aria-label={badge}
              className={`pointer-events-auto absolute rounded-sm transition-colors ${
                isActive
                  ? 'bg-emerald-400/10 ring-2 ring-inset ring-emerald-400/90 hover:bg-emerald-400/20'
                  : 'bg-amber-500/15 outline-2 -outline-offset-2 outline-dashed outline-amber-400 hover:bg-amber-500/25'
              }`}
              data-tooltip={lock.stale_reason ? `${badge} (${lock.stale_reason})` : badge}
              key={lock.lock_id}
              onClick={(event) => {
                event.stopPropagation();
                previewRef.current?.zoomToRegion(normalized.x, normalized.y, normalized.width, normalized.height);
              }}
              onDoubleClick={(event) => event.stopPropagation()}
              onPointerDown={(event) => event.stopPropagation()}
              style={{
                height: `max(${normalized.height * 100}%, 14px)`,
                left: `${normalized.x * 100}%`,
                top: `${normalized.y * 100}%`,
                width: `max(${normalized.width * 100}%, 14px)`,
              }}
              type="button"
            >
              <span
                className={`absolute -top-4 left-0 whitespace-nowrap rounded px-1 py-0.5 text-[9px] font-semibold text-white ${
                  isActive ? 'bg-emerald-600/90' : 'bg-amber-600/95'
                }`}
              >
                {badge}
              </span>
            </button>
          );
        })
      : null;

  const manualReviewsOverlay =
    showLockOverlays && coordinateFrame && manualReviewRegions && manualReviewRegions.length > 0
      ? manualReviewRegions.map((region) => {
          const normalized = worldPolygonToNormalizedBox(region.polygon_world, coordinateFrame);
          const badge = `${region.region_id} · ${region.reason}`;
          return (
            <button
              aria-label={badge}
              className="pointer-events-auto absolute rounded-sm bg-violet-500/10 outline-2 -outline-offset-2 outline-dashed outline-violet-400 transition-colors hover:bg-violet-500/20"
              data-tooltip={badge}
              key={region.region_id}
              onClick={(event) => {
                event.stopPropagation();
                previewRef.current?.zoomToRegion(normalized.x, normalized.y, normalized.width, normalized.height);
              }}
              onDoubleClick={(event) => event.stopPropagation()}
              onPointerDown={(event) => event.stopPropagation()}
              style={{
                height: `max(${normalized.height * 100}%, 14px)`,
                left: `${normalized.x * 100}%`,
                top: `${normalized.y * 100}%`,
                width: `max(${normalized.width * 100}%, 14px)`,
              }}
              type="button"
            >
              <span className="absolute -top-4 left-0 whitespace-nowrap rounded bg-violet-600/90 px-1 py-0.5 text-[9px] font-semibold text-white">
                {region.region_id}
              </span>
            </button>
          );
        })
      : null;

  const draftBoxNormalized =
    dragRect ??
    (draftWorldBox && coordinateFrame
      ? worldPolygonToNormalizedBox(worldBoxToPolygon(draftWorldBox), coordinateFrame)
      : null);

  const draftOverlay = draftBoxNormalized ? (
    <div
      className="pointer-events-none absolute rounded-sm bg-sky-400/15 ring-2 ring-inset ring-sky-400"
      style={{
        height: `max(${draftBoxNormalized.height * 100}%, 8px)`,
        left: `${draftBoxNormalized.x * 100}%`,
        top: `${draftBoxNormalized.y * 100}%`,
        width: `max(${draftBoxNormalized.width * 100}%, 8px)`,
      }}
    />
  ) : null;

  const drawCaptureOverlay =
    isDrawingRegion && coordinateFrame ? (
      <div
        className="pointer-events-auto absolute inset-0 z-20 cursor-crosshair bg-sky-500/5"
        onDoubleClick={(event) => event.stopPropagation()}
        onPointerDown={handleDrawPointerDown}
        onPointerMove={handleDrawPointerMove}
        onPointerUp={handleDrawPointerUp}
      />
    ) : null;

  const combinedOverlay =
    softRegionOverlay ||
    inspectionOverlay ||
    sourceLocksOverlay ||
    manualReviewsOverlay ||
    draftOverlay ||
    drawCaptureOverlay ? (
      <>
        {softRegionOverlay}
        {inspectionOverlay}
        {sourceLocksOverlay}
        {manualReviewsOverlay}
        {draftOverlay}
        {drawCaptureOverlay}
      </>
    ) : null;

  return (
    <ZoomableImagePreview
      alt={alt}
      ariaDescribedBy={hintId}
      detailSrc={detailSrc}
      imageOverlay={combinedOverlay}
      interactionSrc={src}
      labels={{
        fit: t('modals.imageStack.fitPreview'),
        zoomIn: t('modals.imageStack.zoomIn'),
        zoomOut: t('modals.imageStack.zoomOut'),
      }}
      ref={previewRef}
      sourceSize={resultSize}
      toolbarEnd={
        <button
          aria-label={focusLabel}
          className={`flex h-10 w-10 items-center justify-center transition-colors ${
            isFocused ? 'bg-white/15 text-white' : 'text-white/70 hover:bg-white/10 hover:text-white'
          }`}
          data-tooltip={focusLabel}
          onClick={() => onFocusedChange(!isFocused)}
          type="button"
        >
          {isFocused ? <Minimize2 aria-hidden="true" size={17} /> : <Maximize2 aria-hidden="true" size={17} />}
        </button>
      }
    >
      <div className="pointer-events-none absolute top-3 left-3 z-10 flex flex-wrap gap-1.5">
        <span className="rounded-md border border-white/10 bg-black/80 px-2 py-1 text-[11px] font-medium text-white/90">
          {modeLabel}
        </span>
        <span className="rounded-md border border-white/10 bg-black/80 px-2 py-1 text-[11px] text-white/70">
          {alignmentLabel}
        </span>
        {memorySafeScale !== null && (
          <span
            aria-label={`${t('modals.imageStack.memorySafeResult', { percent: memorySafeScale })}${
              fullCanvasLabel ? ` · ${fullCanvasLabel}` : ''
            }`}
            className="pointer-events-auto rounded-md border border-status-warning/35 bg-black/80 px-2 py-1 text-[11px] text-status-warning"
            data-tooltip={fullCanvasLabel}
            role="status"
          >
            {t('modals.imageStack.memorySafeResult', { percent: memorySafeScale })}
          </span>
        )}
        {skippedSources && skippedSources.length > 0 && (
          <span
            aria-label={`${t('modals.imageStack.skippedSources', { number: skippedSources.length })} · ${skippedHint}`}
            className="pointer-events-auto rounded-md border border-status-warning/35 bg-black/80 px-2 py-1 text-[11px] text-status-warning"
            data-tooltip={skippedHint}
            role="status"
          >
            {t('modals.imageStack.skippedSources', { number: skippedSources.length })}
          </span>
        )}
        {stationsUnrelated && (
          <span
            aria-label={`${t('modals.imageStack.stationsUnrelated')} · ${t('modals.imageStack.stationsUnrelatedHint')}`}
            className="pointer-events-auto rounded-md border border-status-warning/35 bg-black/80 px-2 py-1 text-[11px] text-status-warning"
            data-tooltip={t('modals.imageStack.stationsUnrelatedHint')}
            role="status"
          >
            {t('modals.imageStack.stationsUnrelated')}
          </span>
        )}
        {softRegions && (
          <button
            aria-label={`${t('modals.imageStack.softRegions', { number: softRegions.count })} · ${softRegionHint}`}
            aria-pressed={showingSoftRegions}
            className="pointer-events-auto rounded-md bg-black/80 px-2 py-1 text-[11px] text-status-warning ring-1 ring-inset ring-status-warning/35 transition-colors hover:bg-black/90 disabled:cursor-default"
            data-tooltip={softRegionHint}
            disabled={regionFrame === null}
            onClick={() => setShowSoftRegions((shown) => !shown)}
            type="button"
          >
            {showingSoftRegions
              ? t('modals.imageStack.softRegionsHide', { number: softRegions.count })
              : t('modals.imageStack.softRegions', { number: softRegions.count })}
          </button>
        )}
        {coordinateFrame && onToggleDrawRegion && (
          <button
            aria-pressed={isDrawingRegion}
            className={`pointer-events-auto flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium ring-1 ring-inset transition-colors ${
              isDrawingRegion
                ? 'bg-sky-500/85 text-white ring-sky-300'
                : 'bg-black/80 text-white/85 ring-white/20 hover:bg-black/90'
            }`}
            onClick={onToggleDrawRegion}
            type="button"
          >
            <SquareDashedMousePointer aria-hidden="true" size={12} />
            {isDrawingRegion
              ? t('modals.imageStack.inspection.drawingRegion')
              : t('modals.imageStack.inspection.drawRegion')}
          </button>
        )}
        {coordinateFrame && lockRegionCount > 0 && (
          <button
            aria-pressed={showLockOverlays}
            className="pointer-events-auto rounded-md bg-black/80 px-2 py-1 text-[11px] text-emerald-300 ring-1 ring-inset ring-emerald-400/40 transition-colors hover:bg-black/90"
            onClick={() => setShowLockOverlays((shown) => !shown)}
            type="button"
          >
            {showLockOverlays
              ? t('modals.imageStack.inspection.hideLockOverlays', { total: lockRegionCount })
              : t('modals.imageStack.inspection.showLockOverlays', { total: lockRegionCount })}
          </button>
        )}
      </div>

      <p
        className="pointer-events-none absolute bottom-3 left-3 z-10 hidden rounded-md bg-black/75 px-2 py-1 text-[10px] text-white/65 sm:block"
        id={hintId}
      >
        {t('modals.imageStack.previewHint')}
      </p>
    </ZoomableImagePreview>
  );
}

export default memo(ImageStackResultPreview);
