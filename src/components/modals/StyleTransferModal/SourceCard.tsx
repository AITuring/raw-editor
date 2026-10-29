import { ImagePlus, Loader2, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { SourceRole, StyleImage } from './types';

interface SourceCardProps {
  image: StyleImage | null;
  isBusy: boolean;
  onChoose(): void;
  onDrop(event: React.DragEvent<HTMLDivElement>): void;
  onRemove(): void;
  role: SourceRole;
}

export function SourceCard({ image, isBusy, onChoose, onDrop, onRemove, role }: SourceCardProps) {
  const { t } = useTranslation();
  const isReference = role === 'reference';
  const title = isReference
    ? t('styleTransfer.reference', { defaultValue: 'Reference image' })
    : t('styleTransfer.target', { defaultValue: 'Target image' });
  const helper = isReference
    ? t('styleTransfer.referenceHint', { defaultValue: 'The image whose colour language you want to borrow.' })
    : t('styleTransfer.targetHint', { defaultValue: 'The image that keeps its composition and detail.' });

  return (
    <div className="style-transfer-source-card" data-source-role={role}>
      <div className="style-transfer-source-heading">
        <span className="style-transfer-source-index">{isReference ? '01' : '02'}</span>
        <div className="min-w-0">
          <h3>{title}</h3>
          <p>{helper}</p>
        </div>
      </div>
      <div
        className={`style-transfer-dropzone ${image ? 'has-image' : ''} ${isBusy ? 'is-busy' : ''}`}
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
      >
        <button
          aria-busy={isBusy}
          aria-label={image ? t('styleTransfer.replaceImage', { defaultValue: `Replace ${title}` }) : title}
          className="style-transfer-choose-source"
          disabled={isBusy}
          onClick={onChoose}
          type="button"
        >
          {image ? (
            <>
              <span className="style-transfer-image-frame">
                <img alt={image.name} src={image.url} />
              </span>
              <span className="style-transfer-image-meta">
                <span className="style-transfer-image-name" title={image.name}>
                  {image.name}
                </span>
                <span>
                  {image.width.toLocaleString()} × {image.height.toLocaleString()}
                </span>
              </span>
            </>
          ) : (
            <span className="style-transfer-empty-dropzone">
              {isBusy ? (
                <Loader2 aria-hidden="true" className="animate-spin" size={25} />
              ) : (
                <span className="style-transfer-upload-mark" aria-hidden="true">
                  <ImagePlus size={22} />
                </span>
              )}
              <strong>
                {isBusy
                  ? t('styleTransfer.loading', { defaultValue: 'Preparing preview…' })
                  : t('styleTransfer.chooseImage', { defaultValue: 'Choose an image' })}
              </strong>
              <span>{t('styleTransfer.dropHint', { defaultValue: 'or drop it here · JPG, PNG, TIFF, RAW' })}</span>
            </span>
          )}
        </button>
        {image && (
          <button
            aria-label={t('styleTransfer.removeImage', { defaultValue: `Remove ${title}` })}
            className="style-transfer-remove"
            onClick={onRemove}
            type="button"
          >
            <X aria-hidden="true" size={14} />
          </button>
        )}
      </div>
    </div>
  );
}
