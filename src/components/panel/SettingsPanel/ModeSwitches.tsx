import { useMemo } from 'react';
import { Scaling, Image as ImageIcon, Mouse, Touchpad } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';

interface CanvasInputModeSwitchProps {
  mode: 'mouse' | 'trackpad';
  onModeChange: (mode: 'mouse' | 'trackpad') => void;
}

export const CanvasInputModeSwitch = ({ mode, onModeChange }: CanvasInputModeSwitchProps) => {
  const { t } = useTranslation();

  const canvasInputModes = useMemo(
    () => [
      { id: 'mouse', label: t('settings.controls.modes.mouse'), icon: Mouse },
      { id: 'trackpad', label: t('settings.controls.modes.trackpad'), icon: Touchpad },
    ],
    [t],
  );

  return (
    <div className="ui-segmented-frame relative flex w-full">
      {canvasInputModes.map((item) => (
        <button
          key={item.id}
          onClick={() => onModeChange(item.id as 'mouse' | 'trackpad')}
          className={clsx('ui-segmented-option', mode === item.id && 'is-active')}
          style={{ WebkitTapHighlightColor: 'transparent' }}
        >
          {mode === item.id && (
            <motion.span
              layoutId="canvas-input-mode-switch-bubble"
              className="absolute inset-0 z-0 bg-accent"
              style={{ borderRadius: 6 }}
              transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
            />
          )}
          <span className="relative z-10 flex items-center">
            <item.icon size={16} className="mr-2" />
            {item.label}
          </span>
        </button>
      ))}
    </div>
  );
};

interface PreviewModeSwitchProps {
  mode: 'static' | 'dynamic';
  onModeChange: (mode: 'static' | 'dynamic') => void;
}

export const PreviewModeSwitch = ({ mode, onModeChange }: PreviewModeSwitchProps) => {
  const { t } = useTranslation();

  const previewModes = useMemo(
    () => [
      { id: 'static', label: t('settings.processing.modes.static'), icon: ImageIcon },
      { id: 'dynamic', label: t('settings.processing.modes.dynamic'), icon: Scaling },
    ],
    [t],
  );

  return (
    <div className="ui-segmented-frame relative flex w-full">
      {previewModes.map((item) => (
        <button
          key={item.id}
          onClick={() => onModeChange(item.id as 'static' | 'dynamic')}
          className={clsx('ui-segmented-option', mode === item.id && 'is-active')}
          style={{ WebkitTapHighlightColor: 'transparent' }}
        >
          {mode === item.id && (
            <motion.span
              layoutId="preview-mode-switch-bubble"
              className="absolute inset-0 z-0 bg-accent"
              style={{ borderRadius: 6 }}
              transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
            />
          )}
          <span className="relative z-10 flex items-center">
            <item.icon size={16} className="mr-2" />
            {item.label}
          </span>
        </button>
      ))}
    </div>
  );
};
