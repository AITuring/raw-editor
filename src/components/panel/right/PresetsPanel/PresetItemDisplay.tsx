import { useState, useEffect, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Layers, Crop, Wrench, Palette } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import Text from '../../../ui/Text';
import Slider from '../../../ui/Slider';
import { TextColors, TextVariants, TextWeights } from '../../../../types/typography';
import { ADJUSTMENT_GROUPS } from '../../../../utils/adjustments';
import { Preset } from '../../../ui/AppProperties';

interface PresetItemDisplayProps {
  isGeneratingPreviews: boolean;
  preset: Preset;
  previewUrl: string;
  isActive?: boolean;
  intensity?: number;
  onIntensityChange?: (val: number) => void;
  onDragStateChange?: (isDragging: boolean) => void;
}

interface ImageLayer {
  id: string;
  url: string;
  opacity: number;
}

function PreviewImage({ src, alt }: { src: string; alt: string }) {
  const [layers, setLayers] = useState<ImageLayer[]>([]);

  useEffect(() => {
    if (!src) {
      setLayers([]);
      return;
    }

    setLayers((prev) => {
      if (prev.some((l) => l.id === src)) return prev;

      return [...prev, { id: src, url: src, opacity: 0 }];
    });
  }, [src]);

  useEffect(() => {
    const layerToFadeIn = layers.find((l) => l.opacity === 0);
    if (layerToFadeIn) {
      const frame = requestAnimationFrame(() => {
        setLayers((prev) => prev.map((l) => (l.id === layerToFadeIn.id ? { ...l, opacity: 1 } : l)));
      });
      return () => cancelAnimationFrame(frame);
    }
  }, [layers]);

  const handleTransitionEnd = useCallback((finishedId: string) => {
    setLayers((prev) => {
      const finishedIndex = prev.findIndex((l) => l.id === finishedId);
      if (finishedIndex < 0 || prev.length <= 1) return prev;
      return prev.slice(finishedIndex);
    });
  }, []);

  return (
    <div className="absolute inset-0 w-full h-full overflow-hidden rounded-md">
      {layers.map((layer) => (
        <div
          key={layer.id}
          className="absolute inset-0 w-full h-full"
          style={{
            opacity: layer.opacity,
            transition: 'opacity 300ms ease-in-out',
          }}
          onTransitionEnd={() => handleTransitionEnd(layer.id)}
        >
          <img src={layer.url} alt={alt} className="w-full h-full object-cover rounded-md pointer-events-none" />
        </div>
      ))}
    </div>
  );
}

export function PresetItemDisplay({
  preset,
  previewUrl,
  isActive,
  intensity,
  onIntensityChange,
  onDragStateChange,
}: PresetItemDisplayProps) {
  const { t } = useTranslation();
  const geometryKeys = ADJUSTMENT_GROUPS.geometry.flatMap((g) => g.keys);

  const supportsMasks = preset.includeMasks ?? (preset.adjustments?.masks && preset.adjustments.masks.length > 0);
  const supportsGeometry =
    preset.includeCropTransform ?? geometryKeys.some((key) => preset.adjustments?.[key] !== undefined);
  const isTool = preset.presetType === 'tool';
  const tooltipContent = useMemo(() => {
    const features = [];
    if (supportsMasks) features.push(t('editor.presets.supports.masks'));
    if (supportsGeometry) features.push(t('editor.presets.supports.cropTransform'));

    if (features.length === 0) return undefined;
    return t('editor.presets.supports.label', { features: features.join(' + ') });
  }, [supportsMasks, supportsGeometry, t]);

  return (
    <div className="flex flex-col p-2 rounded-lg bg-surface cursor-grabbing">
      <div className="flex items-center gap-3">
        <div
          className="w-20 h-14 bg-bg-tertiary rounded-md flex items-center justify-center shrink-0 relative overflow-hidden"
          data-tooltip={tooltipContent}
        >
          {previewUrl ? (
            <PreviewImage src={previewUrl} alt={`${preset.name} preview`} />
          ) : (
            <Loader2 size={20} className="animate-spin text-status-info" />
          )}

          {(supportsMasks || supportsGeometry) && (
            <>
              <div className="absolute top-0 right-0 w-1/2 h-1/2 bg-linear-to-bl from-black/30 via-black/0 to-transparent pointer-events-none z-0" />

              <div className="absolute top-1 right-1 bg-primary rounded-full px-1.5 py-0.5 flex items-center gap-1.5 backdrop-blur-xs shadow-xs z-10 pointer-events-none">
                {supportsMasks && <Layers size={11} className="text-white" />}
                {supportsGeometry && <Crop size={11} className="text-white" />}
              </div>
            </>
          )}
        </div>

        <div className="grow min-w-0 flex flex-col justify-center">
          <Text color={TextColors.primary} weight={TextWeights.medium} className="truncate">
            {preset.name}
          </Text>
          <div className="flex items-center gap-1.5 mt-0.5">
            {isTool ? (
              <Wrench size={12} className="text-text-secondary" />
            ) : (
              <Palette size={12} className="text-text-secondary" />
            )}
            <Text
              variant={TextVariants.small}
              color={TextColors.secondary}
              className="text-[10px] uppercase tracking-wider"
            >
              {isTool ? t('editor.presets.types.tool') : t('editor.presets.types.style')}
            </Text>
          </div>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {isActive && onIntensityChange && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: 'easeInOut' }}
            className="w-full cursor-auto overflow-hidden"
            onClick={(e: any) => e.stopPropagation()}
            onPointerDown={(e: any) => e.stopPropagation()}
          >
            <div className="mt-3 px-1 pb-1">
              <Slider
                min={0}
                max={200}
                defaultValue={100}
                value={intensity ?? 100}
                onChange={(e: any) => onIntensityChange(Number(e.target.value))}
                onDragStateChange={onDragStateChange}
                label={t('editor.presets.amount')}
                step={1}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
