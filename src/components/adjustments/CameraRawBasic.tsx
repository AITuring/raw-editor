import { useTranslation } from 'react-i18next';
import { Adjustments } from '../../utils/adjustments';
import { AppSettings } from '../ui/AppProperties';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import BasicAdjustments from './Basic';
import ColorPanel from './Color';
import DetailsPanel from './Details';

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
    <div className="space-y-4">
      <div className="p-2 bg-bg-tertiary rounded-md">
        <BasicAdjustments {...sharedProps} variant="toneMapping" />
      </div>

      <ColorPanel
        {...sharedProps}
        variant="whiteBalance"
        isWbPickerActive={isWbPickerActive}
        toggleWbPicker={toggleWbPicker}
      />

      <div className="p-2 bg-bg-tertiary rounded-md">
        <Text variant={TextVariants.heading} className="mb-2">
          {t('adjustments.basic.light')}
        </Text>
        <BasicAdjustments {...sharedProps} variant="light" />
      </div>

      <div className="p-2 bg-bg-tertiary rounded-md">
        <Text variant={TextVariants.heading} className="mb-2">
          {t('adjustments.details.presence')}
        </Text>
        <DetailsPanel {...sharedProps} variant="presenceBare" />
        <ColorPanel {...sharedProps} variant="presenceBare" />
      </div>
    </div>
  );
}
