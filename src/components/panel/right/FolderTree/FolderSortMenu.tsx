import { ChevronUp, ChevronDown, ArrowUpDown, Check } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import Text from '../../../ui/Text';
import { TEXT_COLOR_KEYS, TextColors, TextVariants, TextWeights } from '../../../../types/typography';
import { FolderTreeSort, SortDirection } from '../../../ui/AppProperties';

export function FolderSortMenu({
  sort,
  onChange,
  isOpen,
  setIsOpen,
}: {
  sort: FolderTreeSort;
  onChange: (s: FolderTreeSort) => void;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}) {
  const menuRef = useRef<HTMLDivElement>(null);
  const { t } = useTranslation();

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [setIsOpen]);

  const options = [
    { key: 'name', label: t('library.folders.sort.name') },
    { key: 'created', label: t('library.folders.sort.created') },
    { key: 'modified', label: t('library.folders.sort.modified') },
    { key: 'imageCount', label: t('library.folders.sort.imageCount') },
  ];

  return (
    <div className="relative" ref={menuRef}>
      <button
        className={clsx(
          'ui-surface-button flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-md bg-surface transition-colors hover:bg-card-active',
          isOpen && 'bg-card-active',
        )}
        onClick={() => setIsOpen(!isOpen)}
        data-tooltip={t('library.folders.tooltips.sortFolders')}
      >
        <ArrowUpDown size={16} className="text-text-secondary" />
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1, ease: 'easeOut' }}
            className="absolute right-0 top-full mt-2 w-48 origin-top-right z-50"
          >
            <div className="ui-popover-surface flex flex-col p-1.5">
              <div className="px-3 py-2 relative flex items-center">
                <Text as="div" variant={TextVariants.small} weight={TextWeights.semibold} className="uppercase">
                  {t('library.header.viewOptions.sortBy')}
                </Text>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onChange({
                      ...sort,
                      order:
                        sort.order === SortDirection.Ascending ? SortDirection.Descending : SortDirection.Ascending,
                    });
                  }}
                  data-tooltip={
                    sort.order === SortDirection.Ascending
                      ? t('library.header.viewOptions.sortDescending')
                      : t('library.header.viewOptions.sortAscending')
                  }
                  className="absolute top-1/2 right-3 -translate-y-1/2 rounded-sm bg-transparent p-1 text-text-secondary transition-colors hover:text-text-primary"
                >
                  {sort.order === SortDirection.Ascending ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
              </div>

              {options.map((opt) => {
                const isSelected = sort.key === opt.key;
                return (
                  <button
                    key={opt.key}
                    className={clsx(
                      'flex min-h-8 w-full items-center justify-between rounded-sm px-2.5 py-1.5 text-left text-xs transition-colors duration-150',
                      isSelected ? 'bg-card-active' : 'hover:bg-bg-primary',
                    )}
                    onClick={() => {
                      if (sort.key !== opt.key) {
                        onChange({ key: opt.key as any, order: sort.order });
                      }
                      setIsOpen(false);
                    }}
                  >
                    <Text
                      variant={TextVariants.label}
                      color={TextColors.primary}
                      weight={isSelected ? TextWeights.semibold : TextWeights.normal}
                    >
                      {opt.label}
                    </Text>
                    {isSelected && <Check size={16} className={TEXT_COLOR_KEYS[TextColors.primary]} />}
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
