import { memo, useEffect, useId, useRef, useState } from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ImageStackResultSize, ImageStackSoftRegions } from '../../store/useUIStore';
import { ZoomableImagePreview } from '../preview';
import type { ZoomableImagePreviewHandle } from '../preview/ZoomableImagePreview';

interface ImageStackResultPreviewProps {
  alignmentLabel: string;
  alt: string;
  detailSrc: string | null;
  isFocused: boolean;
  modeLabel: string;
  onFocusedChange(isFocused: boolean): void;
  resultSize: ImageStackResultSize | null;
  softRegions?: ImageStackSoftRegions | null;
  skippedSources?: string[] | null;
  src: string;
  stationsUnrelated?: boolean;
}

function ImageStackResultPreview({
  alignmentLabel,
  alt,
  detailSrc,
  isFocused,
  modeLabel,
  onFocusedChange,
  resultSize,
  softRegions = null,
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
  useEffect(() => setShowSoftRegions(true), [src]);
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
              className={`pointer-events-auto absolute rounded-[50%] border-2 transition-colors ${
                region.kind === 'sources_soft'
                  ? 'border-red-500/90 hover:bg-red-500/15'
                  : 'border-orange-400/90 hover:bg-orange-400/15'
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

  return (
    <ZoomableImagePreview
      alt={alt}
      ariaDescribedBy={hintId}
      detailSrc={detailSrc}
      imageOverlay={softRegionOverlay}
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
            className="pointer-events-auto rounded-md border border-status-warning/35 bg-black/80 px-2 py-1 text-[11px] text-status-warning transition-colors hover:bg-black/90 disabled:cursor-default"
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
