import { useTranslation } from 'react-i18next';
import { Info, Sparkles } from 'lucide-react';
import Slider from '../ui/Slider';
import { Adjustments, DetailsAdjustment } from '../../utils/adjustments';
import { AppSettings } from '../ui/AppProperties';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import { useEditorStore } from '../../store/useEditorStore';
import { useUIStore } from '../../store/useUIStore';
import { AdjustmentSubsection } from '../ui/AdjustmentSubsection';

interface DetailsPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments>): any;
  appSettings: AppSettings | null;
  isForMask?: boolean;
  onDragStateChange?: (isDragging: boolean) => void;
  variant?: 'all' | 'presence' | 'presenceBare' | 'detail' | 'optics';
}

export default function DetailsPanel({
  adjustments,
  setAdjustments,
  appSettings,
  isForMask = false,
  onDragStateChange,
  variant = 'all',
}: DetailsPanelProps) {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((state) => state.selectedImage);
  const setUI = useUIStore((state) => state.setUI);

  const handleAdjustmentChange = (key: string, value: string) => {
    const numericValue = parseInt(value, 10);
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: numericValue }));
  };

  const adjustmentVisibility = appSettings?.adjustmentVisibility || {};
  const showSharpening = variant === 'all' || variant === 'detail';
  const showPresence = variant === 'all' || variant === 'presence' || variant === 'presenceBare';
  const showNoiseReduction = variant === 'all' || variant === 'detail';
  const showChromaticAberration = variant === 'all' || variant === 'optics';

  const openDenoise = () => {
    if (!selectedImage) return;
    setUI({
      denoiseModalState: {
        isOpen: true,
        isProcessing: false,
        previewBase64: null,
        error: null,
        targetPaths: [selectedImage.path],
        progressMessage: null,
        isRaw: selectedImage.isRaw,
      },
    });
  };

  return (
    <div className="camera-raw-section-body">
      {!isForMask && showSharpening && adjustmentVisibility.sharpening !== false && (
        <AdjustmentSubsection title={t('adjustments.details.rawProcessing')}>
          <button className="camera-raw-action-row" disabled={!selectedImage} onClick={openDenoise} type="button">
            <Sparkles aria-hidden="true" size={15} strokeWidth={1.8} />
            <span className="min-w-0 grow">
              <span className="camera-raw-action-title">{t('modals.denoise.titleSingle')}</span>
              <span className="camera-raw-action-description">{t('modals.denoise.description')}</span>
            </span>
          </button>
          <div className="camera-raw-capability-list">
            <div className="camera-raw-capability">
              <span>{t('adjustments.details.rawDetails')}</span>
              <span className={selectedImage?.isRaw ? 'is-available' : ''}>
                {selectedImage?.isRaw ? t('adjustments.details.rawDetailsActive') : t('adjustments.details.rawOnly')}
              </span>
            </div>
            <div className="camera-raw-capability is-disabled" aria-disabled="true">
              <span>{t('adjustments.details.superResolution')}</span>
              <span>{t('adjustments.details.unavailable')}</span>
            </div>
          </div>
        </AdjustmentSubsection>
      )}

      {showSharpening && adjustmentVisibility.sharpening !== false && (
        <AdjustmentSubsection
          title={t('adjustments.details.sharpening')}
          description={
            !isForMask ? (
              <span className="inline-flex items-start gap-1">
                <Info aria-hidden="true" className="mt-0.5 shrink-0" size={12} />
                {t('adjustments.details.sharpeningPipelineNote')}
              </span>
            ) : undefined
          }
        >
          <Slider
            label={t('adjustments.effects.amount')}
            max={100}
            min={isForMask ? -100 : 0}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Sharpness, e.target.value)}
            step={1}
            value={adjustments.sharpness}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.masking')}
            max={80}
            min={0}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.SharpnessThreshold, e.target.value)}
            step={1}
            value={adjustments.sharpnessThreshold ?? 15}
            onDragStateChange={onDragStateChange}
            defaultValue={15}
            fillOrigin="min"
          />
        </AdjustmentSubsection>
      )}

      {showPresence && adjustmentVisibility.presence !== false && (
        <AdjustmentSubsection className={variant === 'presenceBare' ? 'is-bare' : undefined}>
          {variant !== 'presenceBare' && (
            <Text variant={TextVariants.heading} className="mb-2">
              {t('adjustments.details.presence')}
            </Text>
          )}
          <Slider
            label={t('adjustments.details.clarity')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Clarity, e.target.value)}
            step={1}
            value={adjustments.clarity}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.dehaze')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Dehaze, e.target.value)}
            step={1}
            value={adjustments.dehaze}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={variant === 'presenceBare' ? t('adjustments.details.texture') : t('adjustments.details.structure')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Structure, e.target.value)}
            step={1}
            value={adjustments.structure}
            onDragStateChange={onDragStateChange}
          />
          {!isForMask && variant !== 'presenceBare' && (
            <Slider
              label={t('adjustments.details.centre')}
              max={100}
              min={-100}
              onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Centré, e.target.value)}
              step={1}
              value={adjustments.centré}
              onDragStateChange={onDragStateChange}
            />
          )}
        </AdjustmentSubsection>
      )}

      {showNoiseReduction && adjustmentVisibility.noiseReduction !== false && (
        <AdjustmentSubsection
          title={t('adjustments.details.noiseReduction')}
          description={!isForMask ? t('adjustments.details.previewAt100') : undefined}
        >
          <Slider
            label={t('adjustments.details.luminance')}
            max={100}
            min={isForMask ? -100 : 0}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.LumaNoiseReduction, e.target.value)}
            step={1}
            value={adjustments.lumaNoiseReduction}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.color')}
            max={100}
            min={isForMask ? -100 : 0}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.ColorNoiseReduction, e.target.value)}
            step={1}
            value={adjustments.colorNoiseReduction}
            onDragStateChange={onDragStateChange}
          />
        </AdjustmentSubsection>
      )}

      {!isForMask && showChromaticAberration && adjustmentVisibility.chromaticAberration !== false && (
        <AdjustmentSubsection title={t('adjustments.details.chromaticAberration')}>
          <Slider
            label={t('adjustments.details.redCyan')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.ChromaticAberrationRedCyan, e.target.value)}
            step={1}
            value={adjustments.chromaticAberrationRedCyan}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.blueYellow')}
            max={100}
            min={-100}
            onChange={(e: any) =>
              handleAdjustmentChange(DetailsAdjustment.ChromaticAberrationBlueYellow, e.target.value)
            }
            step={1}
            value={adjustments.chromaticAberrationBlueYellow}
            onDragStateChange={onDragStateChange}
          />
        </AdjustmentSubsection>
      )}
    </div>
  );
}
