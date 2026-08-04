import { useTranslation } from 'react-i18next';
import { Adjustments } from '../../utils/adjustments';
import { AppSettings } from '../ui/AppProperties';
import BasicAdjustments from './Basic';
import ColorPanel from './Color';
import { AdjustmentSubsection } from '../ui/AdjustmentSubsection';

interface CameraRawBasicProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments>): any;
  appSettings: AppSettings | null;
  isWbPickerActive?: boolean;
  toggleWbPicker?: () => void;
  onDragStateChange?: (isDragging: boolean) => void;
}

export default function CameraRawBasic({
  adjustments,
  setAdjustments,
  appSettings,
  isWbPickerActive,
  toggleWbPicker,
  onDragStateChange,
}: CameraRawBasicProps) {
  const { t } = useTranslation();
  const sharedProps = { adjustments, setAdjustments, appSettings, onDragStateChange };

  return (
    <div className="camera-raw-section-body">
      {!appSettings?.tonemapperOverrideEnabled && (
        <AdjustmentSubsection>
          <BasicAdjustments {...sharedProps} variant="toneMapping" />
        </AdjustmentSubsection>
      )}

      <AdjustmentSubsection title={t('adjustments.basic.light')}>
        <BasicAdjustments {...sharedProps} variant="light" />
      </AdjustmentSubsection>

      <AdjustmentSubsection>
        <ColorPanel
          {...sharedProps}
          variant="whiteBalance"
          isWbPickerActive={isWbPickerActive}
          toggleWbPicker={toggleWbPicker}
        />
      </AdjustmentSubsection>

      <AdjustmentSubsection title={t('adjustments.color.presence')}>
        <ColorPanel {...sharedProps} variant="presenceBare" />
      </AdjustmentSubsection>
    </div>
  );
}
