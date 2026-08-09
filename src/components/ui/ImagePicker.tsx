import { open } from '@tauri-apps/plugin-dialog';
import { Image as ImageIcon, LoaderCircle, RotateCcw } from 'lucide-react';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import Text from './Text';
import { TextVariants } from '../../types/typography';

interface ImagePickerProps {
  disabled?: boolean;
  imageName: string;
  imageSrc: string;
  isDefault: boolean;
  onImageSelect: (path: string) => Promise<void> | void;
  onUseDefault: () => void;
  label: string;
}

export default function ImagePicker({
  disabled = false,
  imageName,
  imageSrc,
  isDefault,
  onImageSelect,
  onUseDefault,
  label,
}: ImagePickerProps) {
  const { t } = useTranslation();
  const [importError, setImportError] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState(false);

  const handleSelectFile = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [
          {
            name: t('ui.imagePicker.filterLabel'),
            extensions: ['png', 'jpg', 'jpeg', 'webp', 'tif', 'tiff', 'gif'],
          },
        ],
      });
      if (typeof selected === 'string') {
        setImportError(null);
        setIsImporting(true);
        await onImageSelect(selected);
      }
    } catch (err) {
      console.error('Failed to select or import watermark image:', err);
      setImportError(t('export.watermark.importFailed'));
    } finally {
      setIsImporting(false);
    }
  };

  const handleUseDefault = () => {
    setImportError(null);
    onUseDefault();
  };

  const isDisabled = disabled || isImporting;

  return (
    <div className="mb-2">
      <Text variant={TextVariants.label} className="mb-1 select-none">
        {label}
      </Text>
      <div className="flex min-w-0 items-center gap-2">
        <button
          aria-label={t('export.watermark.replaceWatermark')}
          aria-busy={isImporting}
          className="group flex min-h-12 min-w-0 flex-1 touch-manipulation items-center gap-2 rounded-md bg-surface p-1.5 text-left transition-colors hover:bg-card-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-50"
          data-tooltip={t('export.watermark.replaceWatermark')}
          disabled={isDisabled}
          onClick={handleSelectFile}
          type="button"
        >
          <span className="relative flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-sm bg-bg-tertiary p-1">
            {isImporting ? (
              <LoaderCircle
                aria-hidden="true"
                className="animate-spin text-text-secondary motion-reduce:animate-none"
                size={18}
              />
            ) : (
              <>
                <ImageIcon aria-hidden="true" className="absolute text-text-secondary" size={18} />
                {imageSrc && (
                  <img
                    alt=""
                    className="relative size-full bg-bg-tertiary object-contain"
                    key={imageSrc}
                    onError={(event) => {
                      event.currentTarget.style.display = 'none';
                    }}
                    src={imageSrc}
                  />
                )}
              </>
            )}
          </span>
          <span className="min-w-0 flex-1">
            <span className="block truncate text-sm text-text-primary">{imageName}</span>
            <span className="block truncate text-xs text-text-secondary">
              {isDefault ? t('export.watermark.defaultWatermark') : t('export.watermark.customWatermark')}
            </span>
          </span>
        </button>
        {!isDefault && (
          <button
            aria-label={t('export.watermark.useDefaultWatermark')}
            className="flex min-h-11 shrink-0 touch-manipulation items-center gap-1 rounded-md px-2 text-xs text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:cursor-not-allowed disabled:opacity-50"
            data-tooltip={t('export.watermark.useDefaultWatermark')}
            disabled={isDisabled}
            onClick={handleUseDefault}
            type="button"
          >
            <RotateCcw aria-hidden="true" size={14} />
            <span>{t('export.watermark.useDefault')}</span>
          </button>
        )}
      </div>
      {importError && (
        <p aria-live="polite" className="mt-1 text-xs text-text-secondary" role="status">
          {importError}
        </p>
      )}
    </div>
  );
}
