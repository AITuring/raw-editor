import { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { AlertTriangle, ChevronLeft, ChevronRight, LoaderCircle, SquarePen, Star, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import clsx from 'clsx';

import { ZoomableImagePreview } from '../../preview';
import { ImageFile, Invokes } from '../../ui/AppProperties';
import { useProcessStore } from '../../../store/useProcessStore';

interface LibraryQuickPreviewProps {
  image: ImageFile;
  index: number;
  rating: number;
  total: number;
  onClose(): void;
  onEdit(): void;
  onNavigate(direction: -1 | 1): void;
  onRate(rating: number): void;
}

export default function LibraryQuickPreview({
  image,
  index,
  rating,
  total,
  onClose,
  onEdit,
  onNavigate,
  onRate,
}: LibraryQuickPreviewProps) {
  const { t } = useTranslation();
  const thumbnailUrl = useProcessStore((state) => state.thumbnails[image.path]);
  const cachedPreview = useProcessStore((state) => state.previews[image.path]);
  const setPreview = useProcessStore((state) => state.setPreview);
  const [detailSrc, setDetailSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retryRevision, setRetryRevision] = useState(0);

  const fileName = useMemo(() => image.path.split(/[\\/]/).pop()?.split('?vc=')[0] || image.path, [image.path]);
  const sourceSize = useMemo(() => {
    const exif = image.exif ?? {};
    const width = Number(exif.ExifImageWidth || exif.PixelXDimension || 0);
    const height = Number(exif.ExifImageHeight || exif.PixelYDimension || 0);
    return width > 0 && height > 0 ? { width, height } : null;
  }, [image.exif]);

  useEffect(() => {
    const safeThumbnailKey = thumbnailUrl || '';
    const currentPreview = useProcessStore.getState().previews[image.path];
    if (currentPreview && currentPreview.thumbKey === safeThumbnailKey) {
      setDetailSrc(currentPreview.url);
      setIsLoading(false);
      setLoadError(null);
      return;
    }

    let isActive = true;
    setDetailSrc(null);
    const timeout = window.setTimeout(async () => {
      setIsLoading(true);
      setLoadError(null);

      try {
        let adjustments = {};
        try {
          const metadata: any = await invoke(Invokes.LoadMetadata, { path: image.path });
          adjustments = metadata?.adjustments && !metadata.adjustments.is_null ? metadata.adjustments : {};
        } catch (metadataError) {
          console.warn('Quick preview metadata unavailable:', metadataError);
        }

        const bytes = await invoke<Uint8Array>(Invokes.GeneratePreviewForPath, {
          path: image.path,
          jsAdjustments: adjustments,
        });
        if (!isActive) return;

        const blobUrl = URL.createObjectURL(new Blob([new Uint8Array(bytes)], { type: 'image/jpeg' }));
        setPreview(image.path, blobUrl, safeThumbnailKey);
        setDetailSrc(blobUrl);
      } catch (error) {
        if (!isActive) return;
        console.error('Failed to generate quick preview:', error);
        setLoadError(String(error));
      } finally {
        if (isActive) setIsLoading(false);
      }
    }, 120);

    return () => {
      isActive = false;
      window.clearTimeout(timeout);
    };
  }, [cachedPreview?.thumbKey, image.path, retryRevision, setPreview, thumbnailUrl]);

  const interactionSrc = thumbnailUrl || detailSrc;

  return (
    <section
      aria-label={t('library.quickPreview.title')}
      aria-modal="true"
      className="absolute inset-0 z-40 flex min-h-0 flex-col overflow-hidden bg-[#10100f] text-white"
      role="dialog"
    >
      <header className="flex h-12 shrink-0 items-center gap-2 border-b border-white/15 bg-[#181816] px-4">
        <button
          aria-label={t('library.quickPreview.previous')}
          className="flex h-8 w-8 items-center justify-center rounded-sm text-white/70 transition-colors hover:bg-white/10 hover:text-white"
          data-tooltip={t('library.quickPreview.previous')}
          onClick={() => onNavigate(-1)}
          type="button"
        >
          <ChevronLeft aria-hidden="true" size={18} />
        </button>
        <button
          aria-label={t('library.quickPreview.next')}
          className="flex h-8 w-8 items-center justify-center rounded-sm text-white/70 transition-colors hover:bg-white/10 hover:text-white"
          data-tooltip={t('library.quickPreview.next')}
          onClick={() => onNavigate(1)}
          type="button"
        >
          <ChevronRight aria-hidden="true" size={18} />
        </button>

        <div className="min-w-0 flex-1 pl-1">
          <div className="truncate text-xs font-medium text-white/90" title={fileName}>
            {fileName}
          </div>
          <div className="text-[10px] tabular-nums text-white/50">
            {t('library.quickPreview.position', { current: index + 1, total })}
          </div>
        </div>

        <div aria-label={t('library.quickPreview.rating')} className="hidden items-center gap-0.5 sm:flex" role="group">
          {[1, 2, 3, 4, 5].map((starValue) => (
            <button
              aria-label={t('ui.bottomBar.tooltips.rateStars', { count: starValue })}
              className="flex h-7 w-6 items-center justify-center"
              key={starValue}
              onClick={() => onRate(starValue === rating ? 0 : starValue)}
              type="button"
            >
              <Star
                aria-hidden="true"
                className={clsx(
                  'transition-colors',
                  starValue <= rating ? 'fill-accent text-accent' : 'text-white/45 hover:text-accent',
                )}
                size={15}
              />
            </button>
          ))}
        </div>

        <span aria-hidden="true" className="mx-1 h-5 w-px bg-white/10" />
        <button className="ui-button ui-button--primary ui-button--sm" onClick={onEdit} type="button">
          <SquarePen aria-hidden="true" size={15} />
          {t('library.actions.enterEdit')}
        </button>
        <button
          aria-label={t('library.quickPreview.close')}
          className="flex h-8 w-8 items-center justify-center rounded-sm text-white/70 transition-colors hover:bg-white/10 hover:text-white"
          data-tooltip={t('library.quickPreview.close')}
          onClick={onClose}
          type="button"
        >
          <X aria-hidden="true" size={18} />
        </button>
      </header>

      <div className="relative min-h-0 flex-1">
        {interactionSrc ? (
          <ZoomableImagePreview
            alt={fileName}
            className="h-full"
            detailSrc={detailSrc}
            interactionSrc={interactionSrc}
            labels={{
              fit: t('library.quickPreview.fit'),
              toolbar: t('library.quickPreview.zoomTools'),
              zoomIn: t('library.quickPreview.zoomIn'),
              zoomOut: t('library.quickPreview.zoomOut'),
            }}
            sourceSize={sourceSize}
          >
            {isLoading && (
              <div
                aria-live="polite"
                className="absolute left-3 top-3 z-20 flex items-center gap-2 rounded-sm border border-white/10 bg-black/70 px-2.5 py-1.5 text-[11px] text-white/75"
                role="status"
              >
                <LoaderCircle aria-hidden="true" className="animate-spin" size={14} />
                {t('library.quickPreview.loading')}
              </div>
            )}
            {loadError && (
              <div
                className="absolute left-3 top-3 z-20 flex max-w-sm items-center gap-2 rounded-sm border border-status-warning/40 bg-black/80 px-2.5 py-1.5 text-[11px] text-white/80"
                role="status"
              >
                <AlertTriangle aria-hidden="true" className="shrink-0 text-status-warning" size={14} />
                <span className="truncate">{t('library.quickPreview.previewFailed')}</span>
                <button
                  className="ui-inline-action shrink-0"
                  onClick={() => setRetryRevision((revision) => revision + 1)}
                  type="button"
                >
                  {t('library.actions.retry')}
                </button>
              </div>
            )}
          </ZoomableImagePreview>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-white/60">
            {loadError ? (
              <>
                <AlertTriangle aria-hidden="true" className="text-status-warning" size={28} />
                <span className="text-sm">{t('library.quickPreview.previewFailed')}</span>
                <button
                  className="ui-surface-button rounded-sm px-3 py-1.5 text-xs text-white hover:bg-white/10"
                  onClick={() => setRetryRevision((revision) => revision + 1)}
                  type="button"
                >
                  {t('library.actions.retry')}
                </button>
              </>
            ) : (
              <>
                <LoaderCircle aria-hidden="true" className="animate-spin" size={28} />
                <span className="text-sm">{t('library.quickPreview.loading')}</span>
              </>
            )}
          </div>
        )}
      </div>

      <footer className="flex h-9 shrink-0 items-center justify-center border-t border-white/15 bg-[#181816] px-4 text-[10px] text-white/50">
        {t('library.quickPreview.keyboardHint')}
      </footer>
    </section>
  );
}
