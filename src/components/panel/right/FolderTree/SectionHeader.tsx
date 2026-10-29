import { ChevronRight, ChevronDown } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Text from '../../../ui/Text';
import { TextVariants, TextWeights } from '../../../../types/typography';

export function SectionHeader({ title, isOpen, onToggle }: { title: string; isOpen: boolean; onToggle: () => void }) {
  const { t } = useTranslation();

  return (
    <Text
      as="div"
      variant={TextVariants.small}
      weight={TextWeights.bold}
      className="flex items-center w-full px-1 py-1.5 cursor-pointer group"
      onClick={onToggle}
      data-tooltip={
        isOpen
          ? t('library.folders.collapseSection', { section: title })
          : t('library.folders.expandSection', { section: title })
      }
    >
      <div className="p-0.5 rounded-md transition-colors">
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </div>
      <span className="ml-1 uppercase tracking-wider select-none">{title}</span>
    </Text>
  );
}
