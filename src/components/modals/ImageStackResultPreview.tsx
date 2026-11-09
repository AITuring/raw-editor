import { memo, useEffect, useId, useState } from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ImageStackResultSize, ImageStackSoftRegions } from '../../store/useUIStore';
import { ZoomableImagePreview } from '../preview';

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
}: ImageStackResultPreviewProps) {
  const { t } = useTranslation();
  const hintId = useId();
  // The overview with the soft regions circled replaces the result while shown.
  const [showSoftRegions, setShowSoftRegions] = useState(false);
  useEffect(() => setShowSoftRegions(false), [src]);
  const overviewSrc = softRegions?.overviewPath ?? null;
  const showingOverview = showSoftRegions && overviewSrc !== null;
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

  return (
    <ZoomableImagePreview
      alt={alt}
      ariaDescribedBy={hintId}
      detailSrc={showingOverview ? overviewSrc : detailSrc}
      interactionSrc={showingOverview ? overviewSrc : src}
      labels={{
        fit: t('modals.imageStack.fitPreview'),
        zoomIn: t('modals.imageStack.zoomIn'),
        zoomOut: t('modals.imageStack.zoomOut'),
      }}
      sourceSize={showingOverview ? null : resultSize}
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
        {softRegions && (
          <button
            aria-label={`${t('modals.imageStack.softRegions', { number: softRegions.count })} · ${softRegionHint}`}
            aria-pressed={showingOverview}
            className="pointer-events-auto rounded-md border border-status-warning/35 bg-black/80 px-2 py-1 text-[11px] text-status-warning transition-colors hover:bg-black/90 disabled:cursor-default"
            data-tooltip={softRegionHint}
            disabled={overviewSrc === null}
            onClick={() => setShowSoftRegions((shown) => !shown)}
            type="button"
          >
            {showingOverview
              ? t('modals.imageStack.softRegionsBack')
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
