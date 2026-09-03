import { useState, useEffect, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { X, Plus } from 'lucide-react';
import { motion, AnimatePresence, type Variants } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Invokes } from '../components/ui/AppProperties';
import Text from '../components/ui/Text';
import { TextVariants } from '../types/typography';

interface TaggingSubMenuProps {
  paths: string[];
  initialTags: { tag: string; isUser: boolean }[];
  onTagsChanged: (paths: string[], newTags: { tag: string; isUser: boolean }[]) => void;
  appSettings: any;
  hideContextMenu: () => void;
}

const USER_TAG_PREFIX = 'user:';

const tagVariants: Variants = {
  visible: { opacity: 1, scale: 1, transition: { type: 'spring', stiffness: 500, damping: 30 } },
  exit: { opacity: 0, scale: 0.8, transition: { duration: 0.15 } },
};

export default function TaggingSubMenu({
  paths,
  initialTags,
  onTagsChanged,
  appSettings,
  hideContextMenu,
}: TaggingSubMenuProps) {
  const { t } = useTranslation();
  const [tags, setTags] = useState<{ tag: string; isUser: boolean }[]>(initialTags);
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setTags(initialTags);
  }, [initialTags]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleAddTag = async (tagToAdd: string) => {
    const newTagValue = tagToAdd.trim().toLowerCase();
    if (newTagValue && !tags.some((t) => t.tag === newTagValue)) {
      try {
        const prefixedTag = `${USER_TAG_PREFIX}${newTagValue}`;
        await invoke(Invokes.AddTagForPaths, { paths, tag: prefixedTag });
        const newTags = [...tags, { tag: newTagValue, isUser: true }].sort((a, b) => a.tag.localeCompare(b.tag));
        setTags(newTags);
        onTagsChanged(paths, newTags);
        setInputValue('');
      } catch (err) {
        console.error(`Failed to add tag: ${err}`);
      }
    }
  };

  const handleRemoveTag = async (tagToRemove: { tag: string; isUser: boolean }) => {
    try {
      const prefixedTag = tagToRemove.isUser ? `${USER_TAG_PREFIX}${tagToRemove.tag}` : tagToRemove.tag;
      await invoke(Invokes.RemoveTagForPaths, { paths, tag: prefixedTag });
      const newTags = tags.filter((t) => t.tag !== tagToRemove.tag);
      setTags(newTags);
      onTagsChanged(paths, newTags);
    } catch (err) {
      console.error(`Failed to remove tag: ${err}`);
    }
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddTag(inputValue);
    }
    if (e.key === 'Escape') {
      hideContextMenu();
    }
  };

  const shortcuts = appSettings?.taggingShortcuts || [];

  return (
    <div
      className="ui-popover-surface w-64 p-3 text-text-primary"
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <div className="mb-3">
        <div className="flex min-h-8 flex-wrap items-center gap-1 rounded-md border border-border-color bg-bg-primary/45 p-1.5">
          <AnimatePresence>
            {tags.length > 0 ? (
              tags.map((tagItem) => (
                <motion.div
                  key={tagItem.tag}
                  layout
                  variants={tagVariants}
                  initial={false}
                  animate="visible"
                  exit="exit"
                  onClick={() => handleRemoveTag(tagItem)}
                  data-tooltip={t('menus.tagging.removeTooltip', { tag: tagItem.tag })}
                  className="flex items-center gap-1 bg-bg-primary text-text-primary text-xs font-medium px-2 py-1 rounded-sm group cursor-pointer"
                >
                  <span>{tagItem.tag}</span>
                  <span className="rounded-full group-hover:bg-black/20 p-0.5 transition-colors">
                    <X size={12} />
                  </span>
                </motion.div>
              ))
            ) : (
              <motion.span
                key="no-tags-placeholder"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="p-1 select-none"
              >
                <Text variant={TextVariants.small} className="italic">
                  {t('menus.tagging.noTags')}
                </Text>
              </motion.span>
            )}
          </AnimatePresence>
        </div>
      </div>

      <div className="relative mb-3">
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder={t('menus.tagging.placeholder')}
          className="ui-input pr-9"
        />
        <button
          onClick={() => handleAddTag(inputValue)}
          className="absolute right-1 top-1/2 grid h-7 w-7 -translate-y-1/2 place-items-center rounded-sm text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary"
          data-tooltip={t('menus.tagging.addTagTooltip')}
        >
          <Plus size={16} />
        </button>
      </div>

      {shortcuts.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-text-primary px-1 py-1">{t('menus.tagging.shortcutsHeading')}</p>
          <div className="flex flex-wrap gap-1">
            {shortcuts.map((shortcut: string) => (
              <button
                key={shortcut}
                onClick={() => handleAddTag(shortcut)}
                className="bg-surface text-text-secondary hover:bg-card-active hover:text-text-primary text-xs font-medium px-2 py-1 rounded-sm"
              >
                {shortcut}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
