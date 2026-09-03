import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import clsx from 'clsx';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import Switch from '../ui/Switch';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { Preset } from '../ui/AppProperties';
import { ADJUSTMENT_GROUPS } from '../../utils/adjustments';

interface ConfigurePresetModalProps {
  isOpen: boolean;
  onClose(): void;
  onSave(name: string, includeMasks: boolean, includeCropTransform: boolean, presetType: 'tool' | 'style'): void;
  initialPreset?: Preset | null;
}

interface PresetTypeSwitchProps {
  selectedType: 'tool' | 'style';
  onChange: (type: 'tool' | 'style') => void;
}

const PresetTypeSwitch = ({ selectedType, onChange }: PresetTypeSwitchProps) => {
  const { t } = useTranslation();
  const [bubbleStyle, setBubbleStyle] = useState({});
  const isInitialAnimation = useRef(true);

  const presetTypeOptions = useMemo(
    () => [
      {
        id: 'style' as const,
        label: t('modals.configurePreset.typeStyleLabel'),
        title: t('modals.configurePreset.typeStyleDesc'),
      },
      {
        id: 'tool' as const,
        label: t('modals.configurePreset.typeToolLabel'),
        title: t('modals.configurePreset.typeToolDesc'),
      },
    ],
    [t],
  );

  useEffect(() => {
    const selectedIndex = presetTypeOptions.findIndex((m) => m.id === selectedType);
    const safeIndex = selectedIndex >= 0 ? selectedIndex : 0;

    const widthPercent = 100 / presetTypeOptions.length;
    const targetX = `${safeIndex * 100}%`;
    const targetWidth = `${widthPercent}%`;

    if (isInitialAnimation.current) {
      const initialX = selectedType === 'style' ? '-25%' : '100%';

      setBubbleStyle({
        x: [initialX, targetX],
        width: targetWidth,
      });
      isInitialAnimation.current = false;
    } else {
      setBubbleStyle({
        x: targetX,
        width: targetWidth,
      });
    }
  }, [selectedType, presetTypeOptions]);

  return (
    <div className="ui-segmented-frame mt-2 w-full">
      <div className="relative flex w-full">
        <motion.div
          className="absolute top-0 bottom-0 z-0 bg-accent"
          style={{ borderRadius: 4 }}
          animate={bubbleStyle}
          transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
        />
        {presetTypeOptions.map((option) => (
          <button
            key={option.id}
            data-tooltip={option.title}
            onClick={(e) => {
              e.preventDefault();
              onChange(option.id);
            }}
            className={clsx('ui-segmented-option', selectedType === option.id && 'is-active')}
            style={{ WebkitTapHighlightColor: 'transparent' }}
          >
            <span className="relative z-10 flex items-center">{option.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default function ConfigurePresetModal({ isOpen, onClose, onSave, initialPreset }: ConfigurePresetModalProps) {
  const { t } = useTranslation();
  const [name, setName] = useState('');
  const [includeMasks, setIncludeMasks] = useState(false);
  const [includeCropTransform, setIncludeCropTransform] = useState(false);
  const [presetType, setPresetType] = useState<'tool' | 'style'>('style');
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setName(initialPreset?.name || '');
      setIncludeMasks(
        initialPreset?.includeMasks ??
          (initialPreset?.adjustments?.masks && initialPreset.adjustments.masks.length > 0) ??
          false,
      );

      const GEOMETRY_KEYS = ADJUSTMENT_GROUPS.geometry.flatMap((group) => group.keys);
      const hasGeometry =
        initialPreset?.adjustments && Object.keys(initialPreset.adjustments).some((key) => GEOMETRY_KEYS.includes(key));
      setIncludeCropTransform(initialPreset?.includeCropTransform ?? hasGeometry ?? false);

      setPresetType(initialPreset?.presetType || 'style');
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
        setName('');
        setIncludeMasks(false);
        setIncludeCropTransform(false);
        setPresetType('style');
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen, initialPreset]);

  const handleSave = useCallback(() => {
    if (name.trim()) {
      onSave(name.trim(), includeMasks, includeCropTransform, presetType);
      onClose();
    }
  }, [name, includeMasks, includeCropTransform, presetType, onSave, onClose]);

  const handleKeyDown = useCallback(
    (e: any) => {
      if (e.key === 'Enter') {
        handleSave();
      } else if (e.key === 'Escape') {
        onClose();
      }
    },
    [handleSave, onClose],
  );

  if (!isMounted) {
    return null;
  }

  return (
    <div
      className={`app-modal-backdrop ${show ? 'opacity-100' : 'opacity-0'}`}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className={`app-modal-surface app-modal-surface--padded max-w-sm ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
        onClick={(e: any) => e.stopPropagation()}
      >
        <Text variant={TextVariants.heading} className="mb-3">
          {initialPreset ? t('modals.configurePreset.titleConfigure') : t('modals.configurePreset.titleSave')}
        </Text>
        <Input
          autoFocus
          bgClassName="bg-bg-primary"
          onChange={(e: any) => setName(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t('modals.configurePreset.placeholder')}
          type="text"
          value={name}
        />

        <div className="my-5 space-y-3 rounded-md border border-border-color bg-bg-primary/35 p-3">
          <Switch label={t('modals.configurePreset.includeMasks')} checked={includeMasks} onChange={setIncludeMasks} />
          <Switch
            label={t('modals.configurePreset.includeCropTransform')}
            checked={includeCropTransform}
            onChange={setIncludeCropTransform}
          />
        </div>

        <PresetTypeSwitch selectedType={presetType} onChange={setPresetType} />

        <div className="app-modal-actions">
          <Button variant="ghost" onClick={onClose}>
            {t('modals.configurePreset.cancel')}
          </Button>
          <Button disabled={!name.trim()} onClick={handleSave}>
            {t('modals.configurePreset.save')}
          </Button>
        </div>
      </div>
    </div>
  );
}
