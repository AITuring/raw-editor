import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import clsx from 'clsx';
import { ADJUSTMENT_GROUPS, COPYABLE_ADJUSTMENT_KEYS, CopyPasteSettings, PasteMode } from '../../utils/adjustments';
import Button from '../ui/Button';
import Switch from '../ui/Switch';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';

interface CopyPasteSettingsModalProps {
  isOpen: boolean;
  onClose(): void;
  onSave(settings: CopyPasteSettings): void;
  settings: CopyPasteSettings;
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const DEFAULT_SETTINGS: CopyPasteSettings = {
  mode: PasteMode.Merge,
  includedAdjustments: COPYABLE_ADJUSTMENT_KEYS,
  knownAdjustments: [],
  autoSync: false,
};

interface PasteModeSwitchProps {
  selectedMode: PasteMode;
  onModeChange: (mode: PasteMode) => void;
  isVisible: boolean;
}

const PasteModeSwitch = ({ selectedMode, onModeChange, isVisible }: PasteModeSwitchProps) => {
  const { t } = useTranslation();
  const [buttonRefs, setButtonRefs] = useState<Map<string, HTMLButtonElement>>(new Map());
  const [bubbleStyle, setBubbleStyle] = useState({});
  const containerRef = useRef<HTMLDivElement>(null);
  const isInitialAnimation = useRef(true);

  const pasteModeOptions = useMemo(
    () => [
      { id: PasteMode.Merge, label: t('modals.copyPaste.modeMerge') },
      { id: PasteMode.Replace, label: t('modals.copyPaste.modeReplace') },
    ],
    [t],
  );

  useEffect(() => {
    const selectedButton = buttonRefs.get(selectedMode);

    if (!isVisible || !selectedButton || !containerRef.current) {
      return;
    }

    const targetStyle = {
      x: selectedButton.offsetLeft,
      width: selectedButton.offsetWidth,
    };

    if (isInitialAnimation.current && containerRef.current.offsetWidth > 0) {
      let initialX;
      if (selectedMode === PasteMode.Replace) {
        initialX = containerRef.current.offsetWidth;
      } else {
        initialX = -targetStyle.width;
      }

      setBubbleStyle({
        x: [initialX, targetStyle.x],
        width: targetStyle.width,
      });
      isInitialAnimation.current = false;
    } else {
      setBubbleStyle(targetStyle);
    }
  }, [selectedMode, buttonRefs, isVisible]);

  useEffect(() => {
    if (!isVisible) {
      isInitialAnimation.current = true;
    }
  }, [isVisible]);

  return (
    <div ref={containerRef} className="copy-paste-mode-control">
      <motion.div
        animate={bubbleStyle}
        className="copy-paste-mode-indicator"
        transition={{ duration: 0.16, ease: [0.22, 1, 0.36, 1] }}
      />
      {pasteModeOptions.map((option) => (
        <button
          aria-pressed={selectedMode === option.id}
          className={clsx('copy-paste-mode-option', selectedMode === option.id && 'is-active')}
          key={option.id}
          onClick={() => onModeChange(option.id)}
          ref={(el) => {
            if (el) {
              const newRefs = new Map(buttonRefs);
              if (newRefs.get(option.id) !== el) {
                newRefs.set(option.id, el);
                setButtonRefs(newRefs);
              }
            }
          }}
          style={{ WebkitTapHighlightColor: 'transparent' }}
          type="button"
        >
          <span>{option.label}</span>
        </button>
      ))}
    </div>
  );
};

export default function CopyPasteSettingsModal({ isOpen, onClose, onSave, settings }: CopyPasteSettingsModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [localSettings, setLocalSettings] = useState<CopyPasteSettings>(settings || DEFAULT_SETTINGS);

  useEffect(() => {
    if (isOpen) {
      setLocalSettings(settings || DEFAULT_SETTINGS);
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => setIsMounted(false), 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen, settings]);

  const handleSave = useCallback(() => {
    onSave(localSettings);
    onClose();
  }, [localSettings, onSave, onClose]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    },
    [onClose],
  );

  useEffect(() => {
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, handleKeyDown]);

  const handleSelectAll = () => {
    setLocalSettings((prev) => ({ ...prev, includedAdjustments: [...COPYABLE_ADJUSTMENT_KEYS] }));
  };

  const handleSelectNone = () => {
    setLocalSettings((prev) => ({ ...prev, includedAdjustments: [] }));
  };

  const handleGroupToggle = (keys: string[], checked: boolean) => {
    setLocalSettings((prev) => {
      const newSet = new Set(prev.includedAdjustments);
      keys.forEach((key) => {
        if (checked) newSet.add(key);
        else newSet.delete(key);
      });
      return { ...prev, includedAdjustments: Array.from(newSet) };
    });
  };

  if (!isMounted) return null;

  return (
    <div className={`app-modal-backdrop ${show ? 'opacity-100' : 'opacity-0'}`} onClick={onClose}>
      <div
        aria-labelledby="copy-paste-settings-title"
        aria-modal="true"
        className={`app-modal-surface app-modal-surface--structured copy-paste-dialog ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
      >
        <header className="copy-paste-dialog-header">
          <Text className="copy-paste-dialog-title" id="copy-paste-settings-title" variant={TextVariants.heading}>
            {t('modals.copyPaste.title')}
          </Text>
        </header>

        <div className="copy-paste-dialog-body custom-scrollbar">
          <section className="copy-paste-section">
            <Text className="copy-paste-section-title" variant={TextVariants.heading}>
              {t('modals.copyPaste.pasteMode')}
            </Text>
            <PasteModeSwitch
              isVisible={show}
              onModeChange={(mode) => setLocalSettings((p) => ({ ...p, mode }))}
              selectedMode={localSettings.mode}
            />
            <Text className="copy-paste-description" variant={TextVariants.small}>
              <b>{t('modals.copyPaste.modeMerge')}:</b> {t('modals.copyPaste.descMerge')}
              <br />
              <b>{t('modals.copyPaste.modeReplace')}:</b> {t('modals.copyPaste.descReplace')}
            </Text>
          </section>

          <section className="copy-paste-section">
            <Text className="copy-paste-section-title" variant={TextVariants.heading}>
              {t('modals.copyPaste.autoSyncTitle')}
            </Text>
            <Switch
              checked={localSettings.autoSync}
              className="copy-paste-auto-sync"
              label={t('modals.copyPaste.autoSyncLabel')}
              onChange={(checked) => setLocalSettings((p) => ({ ...p, autoSync: checked }))}
            />
            <Text className="copy-paste-description" variant={TextVariants.small}>
              {t('modals.copyPaste.autoSyncDesc')}
            </Text>
          </section>

          <section className="copy-paste-section copy-paste-adjustments-section">
            <div className="copy-paste-adjustments-heading">
              <Text className="copy-paste-section-title" variant={TextVariants.heading}>
                {t('modals.copyPaste.includedAdjustments')}
              </Text>
              <div className="copy-paste-selection-actions">
                <Button variant="secondary" size="sm" onClick={handleSelectAll}>
                  {t('modals.copyPaste.selectAll')}
                </Button>
                <Button variant="secondary" size="sm" onClick={handleSelectNone}>
                  {t('modals.copyPaste.selectNone')}
                </Button>
              </div>
            </div>
            <div className="copy-paste-adjustment-list custom-scrollbar">
              <div className="copy-paste-adjustment-grid">
                {Object.entries(ADJUSTMENT_GROUPS).map(([section, groups]) => (
                  <section className="copy-paste-adjustment-group" key={section}>
                    <Text className="copy-paste-adjustment-group-title" variant={TextVariants.heading}>
                      {t(`editor.adjustments.sections.${section}`, { defaultValue: capitalize(section) })}
                    </Text>
                    <div className="copy-paste-adjustment-options">
                      {groups.map((group) => {
                        const isFullyChecked = group.keys.every((key) =>
                          localSettings.includedAdjustments.includes(key),
                        );

                        return (
                          <Switch
                            checked={isFullyChecked}
                            key={group.label}
                            label={t(group.label as never) as string}
                            onChange={(checked) => handleGroupToggle(group.keys, checked)}
                          />
                        );
                      })}
                    </div>
                  </section>
                ))}
              </div>
            </div>
          </section>
        </div>

        <footer className="copy-paste-dialog-footer">
          <Button variant="secondary" onClick={onClose}>
            {t('modals.copyPaste.cancel')}
          </Button>
          <Button onClick={handleSave}>{t('modals.copyPaste.save')}</Button>
        </footer>
      </div>
    </div>
  );
}
