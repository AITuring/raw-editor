import { useTranslation } from 'react-i18next';

import Text from '../../../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../../../types/typography';

export const ConnectionStatus = () => {
  const { t } = useTranslation();

  return (
    <div className="bg-surface rounded-lg px-4 py-2 flex items-center gap-2">
      <Text variant={TextVariants.label}>{t('editor.ai.connection.builtinLabel')}</Text>
      <Text
        aria-live="polite"
        as="span"
        color={TextColors.success}
        className="semantic-status"
        data-tone="success"
        role="status"
        variant={TextVariants.label}
        weight={TextWeights.medium}
      >
        <span aria-hidden="true" className="semantic-status__dot" />
        {t('editor.ai.connection.ready')}
      </Text>
    </div>
  );
};
