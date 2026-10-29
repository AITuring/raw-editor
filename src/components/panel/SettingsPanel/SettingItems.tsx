import Button from '../../ui/Button';
import { useTranslation } from 'react-i18next';
import Text from '../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';

interface DataActionItemProps {
  buttonAction(): void;
  buttonText: string;
  description: any;
  disabled?: boolean;
  icon: any;
  isProcessing: boolean;
  message: string;
  title: string;
}

interface SettingItemProps {
  children: any;
  description?: string;
  label: string;
}

export const SettingItem = ({ children, description, label }: SettingItemProps) => (
  <div className="settings-item">
    <Text variant={TextVariants.label} weight={TextWeights.semibold} color={TextColors.primary} className="block">
      {label}
    </Text>
    {children}
    {description && (
      <Text variant={TextVariants.small} className="settings-item-description">
        {description}
      </Text>
    )}
  </div>
);

export const DataActionItem = ({
  buttonAction,
  buttonText,
  description,
  disabled = false,
  icon,
  isProcessing,
  message,
  title,
}: DataActionItemProps) => {
  const { t } = useTranslation();

  return (
    <div className="pb-8 border-b border-border-color last:border-b-0 last:pb-0">
      <Text variant={TextVariants.label} weight={TextWeights.semibold} color={TextColors.primary} className="mb-2">
        {title}
      </Text>
      <Text variant={TextVariants.small} className="mb-3">
        {description}
      </Text>
      <Button variant="destructive" onClick={buttonAction} disabled={isProcessing || disabled}>
        {icon}
        {isProcessing ? t('settings.data.statuses.processing') : buttonText}
      </Button>
      {message && (
        <Text color={TextColors.accent} className="mt-3">
          {message}
        </Text>
      )}
    </div>
  );
};
