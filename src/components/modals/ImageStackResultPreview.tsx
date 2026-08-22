import { memo, useId } from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { ImageStackResultSize } from '../../store/useUIStore';
import { ZoomableImagePreview } from '../preview';

interface ImageStackResultPreviewProps {
  alignmentLabel: string;
  alt: string;
  detailSrc: string | null;
  isFocused: boolean;
  modeLabel: string;
  onFocusedChange(isFocused: boolean): void;
  resultSize: ImageStackResultSize | null;
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
  src,
}: ImageStackResultPreviewProps) {
  const { t } = useTranslation();
  const hintId = useId();
  const focusLabel = isFocused ? t('modals.imageStack.exitFocusPreview') : t('modals.imageStack.focusPreview');

  return (
    <ZoomableImagePreview
      alt={alt}
      ariaDescribedBy={hintId}
      detailSrc={detailSrc}
      interactionSrc={src}
      labels={{
        fit: t('modals.imageStack.fitPreview'),
        zoomIn: t('modals.imageStack.zoomIn'),
        zoomOut: t('modals.imageStack.zoomOut'),
      }}
      sourceSize={resultSize}
      toolbarEnd={
        <button
          aria-label={focusLabel}
          className={`flex h-10 w-10 items-center justify-center border-l border-white/10 transition-colors ${
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
