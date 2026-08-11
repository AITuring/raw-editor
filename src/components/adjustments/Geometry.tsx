import { useTranslation } from 'react-i18next';
import clsx from 'clsx';
import { Adjustments, TransformAdjustment } from '../../utils/adjustments';
import Slider from '../ui/Slider';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';

interface GeometryPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((previous: Adjustments) => Adjustments)): void;
  onDragStateChange?: (isDragging: boolean) => void;
  onTransformChange?: () => void;
  compact?: boolean;
}

export default function GeometryPanel({
  adjustments,
  setAdjustments,
  onDragStateChange,
  onTransformChange,
  compact = false,
}: GeometryPanelProps) {
  const { t } = useTranslation();

  const handleChange = (key: TransformAdjustment, value: string | number) => {
    onTransformChange?.();
    setAdjustments((previous) => ({ ...previous, [key]: Number(value) }));
  };

  return (
    <div className={clsx(compact ? 'crop-geometry-controls' : 'space-y-4')}>
      <div className={clsx(compact ? 'crop-geometry-group' : 'p-2 bg-bg-tertiary rounded-md')}>
        <Text variant={TextVariants.heading} className="mb-2">
          {t('modals.transform.distortion')}
        </Text>
        <Slider
          label={t('modals.transform.amount')}
          value={adjustments.transformDistortion}
          min={-100}
          max={100}
          defaultValue={0}
          step={1}
          onChange={(e) => handleChange(TransformAdjustment.TransformDistortion, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
      </div>

      <div className={clsx(compact ? 'crop-geometry-group' : 'p-2 bg-bg-tertiary rounded-md')}>
        <Text variant={TextVariants.heading} className="mb-2">
          {t('modals.transform.perspective')}
        </Text>
        <Slider
          label={t('modals.transform.vertical')}
          value={adjustments.transformVertical}
          min={-100}
          max={100}
          defaultValue={0}
          step={1}
          onChange={(e) => handleChange(TransformAdjustment.TransformVertical, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
        <Slider
          label={t('modals.transform.horizontal')}
          value={adjustments.transformHorizontal}
          min={-100}
          max={100}
          defaultValue={0}
          step={1}
          onChange={(e) => handleChange(TransformAdjustment.TransformHorizontal, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
      </div>

      <div className={clsx(compact ? 'crop-geometry-group' : 'p-2 bg-bg-tertiary rounded-md')}>
        <Text variant={TextVariants.heading} className="mb-2">
          {t('modals.transform.title')}
        </Text>
        <Slider
          label={t('modals.transform.rotate')}
          value={adjustments.transformRotate}
          min={-45}
          max={45}
          defaultValue={0}
          step={0.1}
          onChange={(e) => handleChange(TransformAdjustment.TransformRotate, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
        <Slider
          label={t('modals.transform.aspect')}
          value={adjustments.transformAspect}
          min={-100}
          max={100}
          defaultValue={0}
          step={1}
          onChange={(e) => handleChange(TransformAdjustment.TransformAspect, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
        <Slider
          label={t('modals.transform.scale')}
          value={adjustments.transformScale}
          min={50}
          max={150}
          defaultValue={100}
          step={1}
          onChange={(e) => handleChange(TransformAdjustment.TransformScale, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
      </div>

      <div className={clsx(compact ? 'crop-geometry-group' : 'p-2 bg-bg-tertiary rounded-md')}>
        <Text variant={TextVariants.heading} className="mb-2">
          {t('modals.transform.offset')}
        </Text>
        <Slider
          label={t('modals.transform.xAxis')}
          value={adjustments.transformXOffset}
          min={-100}
          max={100}
          defaultValue={0}
          step={1}
          onChange={(e) => handleChange(TransformAdjustment.TransformXOffset, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
        <Slider
          label={t('modals.transform.yAxis')}
          value={adjustments.transformYOffset}
          min={-100}
          max={100}
          defaultValue={0}
          step={1}
          onChange={(e) => handleChange(TransformAdjustment.TransformYOffset, e.target.value)}
          onDragStateChange={onDragStateChange}
        />
      </div>
    </div>
  );
}
