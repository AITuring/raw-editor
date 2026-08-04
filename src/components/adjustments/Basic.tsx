import { useTranslation } from 'react-i18next';
import { Adjustments, BasicAdjustment } from '../../utils/adjustments';
import Slider from '../ui/Slider';

interface BasicAdjustmentsProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments>): any;
  isForMask?: boolean;
  onDragStateChange?: (isDragging: boolean) => void;
  appSettings?: any;
  variant?: 'all' | 'toneMapping' | 'light';
}

export default function BasicAdjustments({
  adjustments,
  setAdjustments,
  isForMask = false,
  onDragStateChange,
  appSettings,
  variant = 'all',
}: BasicAdjustmentsProps) {
  const { t } = useTranslation();

  const handleAdjustmentChange = (key: BasicAdjustment, value: string | number) => {
    const numericValue = Number.parseFloat(String(value));
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: numericValue }));
  };

  const hideToneMapper = isForMask || appSettings?.tonemapperOverrideEnabled;
  const showToneMapping = variant === 'all' || variant === 'toneMapping';
  const showLight = variant === 'all' || variant === 'light';

  return (
    <div>
      {showToneMapping && !hideToneMapper && (
        <label className="camera-raw-field">
          <span className="camera-raw-field-label">{t('adjustments.basic.toneMapper')}</span>
          <select
            aria-label={t('adjustments.basic.toneMapper')}
            className="camera-raw-select"
            onChange={(event) =>
              setAdjustments((prev: Partial<Adjustments>) => ({
                ...prev,
                toneMapper: event.target.value as 'basic' | 'agx',
              }))
            }
            value={adjustments.toneMapper || 'basic'}
          >
            <option value="basic">{t('adjustments.basic.mappers.basic')}</option>
            <option value="agx">{t('adjustments.basic.mappers.agx')}</option>
          </select>
        </label>
      )}

      {showLight && (
        <>
          <Slider
            label={t('adjustments.basic.exposure')}
            max={5}
            min={-5}
            onChange={(event) => handleAdjustmentChange(BasicAdjustment.Exposure, event.target.value)}
            step={0.01}
            value={adjustments.exposure}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.basic.brightness')}
            max={5}
            min={-5}
            onChange={(event) => handleAdjustmentChange(BasicAdjustment.Brightness, event.target.value)}
            step={0.01}
            value={adjustments.brightness}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.basic.contrast')}
            max={100}
            min={-100}
            onChange={(event) => handleAdjustmentChange(BasicAdjustment.Contrast, event.target.value)}
            step={1}
            value={adjustments.contrast}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.basic.highlights')}
            max={100}
            min={-100}
            onChange={(event) => handleAdjustmentChange(BasicAdjustment.Highlights, event.target.value)}
            step={1}
            value={adjustments.highlights}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.basic.shadows')}
            max={100}
            min={-100}
            onChange={(event) => handleAdjustmentChange(BasicAdjustment.Shadows, event.target.value)}
            step={1}
            value={adjustments.shadows}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.basic.whites')}
            max={100}
            min={-100}
            onChange={(event) => handleAdjustmentChange(BasicAdjustment.Whites, event.target.value)}
            step={1}
            value={adjustments.whites}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.basic.blacks')}
            max={100}
            min={-100}
            onChange={(event) => handleAdjustmentChange(BasicAdjustment.Blacks, event.target.value)}
            step={1}
            value={adjustments.blacks}
            onDragStateChange={onDragStateChange}
          />
        </>
      )}
    </div>
  );
}
