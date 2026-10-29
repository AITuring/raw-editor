import { useTranslation } from 'react-i18next';
import { WatermarkAnchor } from '../../../ui/ExportImportProperties';
import Text from '../../../ui/Text';
import { TextVariants } from '../../../../types/typography';

export function WatermarkPreview({
  anchor,
  scale,
  spacing,
  opacity,
  watermarkImageSrc,
  baseImageSrc,
  imageAspectRatio,
}: {
  anchor: WatermarkAnchor;
  scale: number;
  spacing: number;
  opacity: number;
  watermarkImageSrc: string;
  baseImageSrc: string | null;
  imageAspectRatio: number;
}) {
  const { t } = useTranslation();

  const getPositionStyles = () => {
    const minDimPercent = imageAspectRatio > 1 ? 100 / imageAspectRatio : 100;
    const watermarkSizePercent = minDimPercent * (scale / 100);
    const spacingPercent = minDimPercent * (spacing / 100);

    const styles: React.CSSProperties = {
      width: `${watermarkSizePercent}%`,
      opacity: opacity / 100,
      position: 'absolute',
    };

    const spacingString = `${spacingPercent}%`;

    switch (anchor) {
      case WatermarkAnchor.TopLeft:
        styles.top = spacingString;
        styles.left = spacingString;
        break;
      case WatermarkAnchor.TopCenter:
        styles.top = spacingString;
        styles.left = '50%';
        styles.transform = 'translateX(-50%)';
        break;
      case WatermarkAnchor.TopRight:
        styles.top = spacingString;
        styles.right = spacingString;
        break;
      case WatermarkAnchor.CenterLeft:
        styles.top = '50%';
        styles.left = spacingString;
        styles.transform = 'translateY(-50%)';
        break;
      case WatermarkAnchor.Center:
        styles.top = '50%';
        styles.left = '50%';
        styles.transform = 'translate(-50%, -50%)';
        break;
      case WatermarkAnchor.CenterRight:
        styles.top = '50%';
        styles.right = spacingString;
        styles.transform = 'translateY(-50%)';
        break;
      case WatermarkAnchor.BottomLeft:
        styles.bottom = spacingString;
        styles.left = spacingString;
        break;
      case WatermarkAnchor.BottomCenter:
        styles.bottom = spacingString;
        styles.left = '50%';
        styles.transform = 'translateX(-50%)';
        break;
      case WatermarkAnchor.BottomRight:
        styles.bottom = spacingString;
        styles.right = spacingString;
        break;
    }
    return styles;
  };

  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-border-color bg-surface"
      style={{ aspectRatio: imageAspectRatio }}
    >
      <div className="absolute inset-0 flex items-center justify-center">
        <Text variant={TextVariants.label}>{t('export.watermark.previewText')}</Text>
      </div>
      {baseImageSrc && (
        <img
          alt=""
          className="absolute inset-0 size-full object-cover"
          key={baseImageSrc}
          onError={(event) => {
            event.currentTarget.style.display = 'none';
          }}
          src={baseImageSrc}
        />
      )}
      <div style={getPositionStyles()}>
        <div className="flex w-full items-center justify-center">
          <img
            alt=""
            className="block h-auto w-full select-none object-contain"
            key={watermarkImageSrc}
            onError={(event) => {
              event.currentTarget.style.display = 'none';
            }}
            src={watermarkImageSrc}
          />
        </div>
      </div>
    </div>
  );
}
