import { useState, useEffect, useRef, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'react-toastify';
import { Loader2, Circle, Hexagon, Octagon, Aperture } from 'lucide-react';
import { motion } from 'framer-motion';
import clsx from 'clsx';
import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, Effect, CreativeAdjustment, DetailsAdjustment } from '../../utils/adjustments';
import LUTControl from '../ui/LUTControl';
import { AppSettings } from '../ui/AppProperties';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import { DepthRangePicker } from '../ui/DepthRangePicker';
import { useProcessStore } from '../../store/useProcessStore';
import DetailsPanel from './Details';
import { AdjustmentSubsection } from '../ui/AdjustmentSubsection';

interface EffectsPanelProps {
  adjustments: Adjustments;
  isForMask?: boolean;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Adjustments)): any;
  handleLutSelect(path: string): void;
  onLutHover?: (path: string | null) => void;
  appSettings: AppSettings | null;
  onDragStateChange?: (isDragging: boolean) => void;
  variant?: 'all' | 'effects' | 'lensBlur';
}

interface BokehShapeSwitchProps {
  selectedShape: string;
  onShapeChange: (shape: string) => void;
}

const BokehShapeSwitch = ({ selectedShape, onShapeChange }: BokehShapeSwitchProps) => {
  const { t } = useTranslation();
  const [bubbleStyle, setBubbleStyle] = useState({});
  const [isLabelHovered, setIsLabelHovered] = useState(false);
  const isInitialAnimation = useRef(true);

  const shapeOptions = useMemo(
    () => [
      { id: 'circle', icon: Circle, title: t('adjustments.effects.bokehCircular') },
      { id: 'hexagon', icon: Hexagon, title: t('adjustments.effects.bokehHexagonal') },
      { id: 'octagon', icon: Octagon, title: t('adjustments.effects.bokehOctagonal') },
      { id: 'ring', icon: Aperture, title: t('adjustments.effects.bokehRing') },
    ],
    [t],
  );

  useEffect(() => {
    const selectedIndex = shapeOptions.findIndex((m) => m.id === selectedShape);
    const safeIndex = selectedIndex >= 0 ? selectedIndex : 0;

    const widthPercent = 100 / shapeOptions.length;
    const targetX = `${safeIndex * 100}%`;
    const targetWidth = `${widthPercent}%`;

    if (isInitialAnimation.current) {
      setBubbleStyle({
        x: ['-25%', targetX],
        width: targetWidth,
      });
      isInitialAnimation.current = false;
    } else {
      setBubbleStyle({
        x: targetX,
        width: targetWidth,
      });
    }
  }, [selectedShape, shapeOptions]);

  const handleReset = () => {
    onShapeChange('circle');
  };

  return (
    <div className="flex flex-col gap-2 mt-3">
      <button
        aria-label={`${t('adjustments.effects.bokehShape')}: ${t('ui.slider.reset')}`}
        className="grid min-h-6 w-fit cursor-pointer items-center"
        onClick={handleReset}
        onMouseEnter={() => setIsLabelHovered(true)}
        onMouseLeave={() => setIsLabelHovered(false)}
        type="button"
      >
        <Text
          variant={TextVariants.label}
          aria-hidden={isLabelHovered}
          className={`col-start-1 row-start-1 text-text-secondary select-none transition-opacity duration-200 ease-in-out ${
            isLabelHovered ? 'opacity-0' : 'opacity-100'
          }`}
        >
          {t('adjustments.effects.bokehShape')}
        </Text>
        <Text
          variant={TextVariants.label}
          aria-hidden={!isLabelHovered}
          className={`col-start-1 row-start-1 text-accent! select-none transition-opacity duration-200 ease-in-out pointer-events-none ${
            isLabelHovered ? 'opacity-100' : 'opacity-0'
          }`}
        >
          {t('ui.slider.reset')}
        </Text>
      </button>

      <div className="w-full p-1 bg-bg-primary rounded-md">
        <div className="relative flex w-full">
          <motion.div
            className="absolute top-0 bottom-0 z-0 bg-accent"
            style={{ borderRadius: 6 }}
            animate={bubbleStyle}
            transition={{ duration: 0.15, ease: [0.22, 1, 0.36, 1] }}
          />
          {shapeOptions.map((shape) => {
            const Icon = shape.icon;
            return (
              <button
                aria-label={shape.title}
                aria-pressed={selectedShape === shape.id}
                key={shape.id}
                data-tooltip={shape.title}
                onClick={() => onShapeChange(shape.id)}
                className={clsx(
                  'relative flex-1 flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
                  {
                    'text-text-secondary hover:text-text-primary hover:bg-surface': selectedShape !== shape.id,
                    'text-button-text': selectedShape === shape.id,
                  },
                )}
                style={{ WebkitTapHighlightColor: 'transparent' }}
                type="button"
              >
                <span className="relative z-10 flex items-center">
                  <Icon size={16} strokeWidth={2} />
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default function EffectsPanel({
  adjustments,
  setAdjustments,
  isForMask = false,
  handleLutSelect,
  onLutHover,
  appSettings,
  onDragStateChange,
  variant = 'all',
}: EffectsPanelProps) {
  const { t } = useTranslation();
  const [isGeneratingDepth, setIsGeneratingDepth] = useState(false);
  const aiModelDownloadStatus = useProcessStore((state) => state.aiModelDownloadStatus);

  const handleGenerateLensBlurDepthMap = async () => {
    setIsGeneratingDepth(true);
    try {
      const b64: string = await invoke('generate_full_image_depth_map', { jsAdjustments: adjustments });
      setAdjustments((prev: Partial<Adjustments>) => ({
        ...prev,
        lensBlurDepthMap: b64,
      }));
    } catch (e: any) {
      toast.error(`Failed to generate depth map: ${e}`);
      setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, lensBlurEnabled: false }));
    } finally {
      setIsGeneratingDepth(false);
    }
  };

  const handleAdjustmentChange = (key: string, value: any) => {
    const numericValue = typeof value === 'boolean' ? value : parseInt(value, 10);
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: numericValue }));
  };

  const handleLutIntensityChange = (intensity: number) => {
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, lutIntensity: intensity }));
  };

  const handleLutClear = () => {
    setAdjustments((prev: Partial<Adjustments>) => ({
      ...prev,
      lutPath: null,
      lutName: null,
      lutData: null,
      lutSize: 0,
      lutIntensity: 100,
    }));
  };

  const handleLensBlurToggle = (enabled: boolean) => {
    handleAdjustmentChange(Effect.LensBlurEnabled, enabled);
    if (enabled && !adjustments.lensBlurDepthMap) {
      handleGenerateLensBlurDepthMap();
    }
  };

  const adjustmentVisibility = appSettings?.adjustmentVisibility || {};
  const showEffects = variant === 'all' || variant === 'effects';
  const showLensBlur = variant === 'all' || variant === 'lensBlur';

  return (
    <div className="camera-raw-section-body">
      {showEffects && (
        <>
          <AdjustmentSubsection title={t('adjustments.details.presence')}>
            <DetailsPanel
              adjustments={adjustments}
              setAdjustments={setAdjustments}
              appSettings={appSettings}
              isForMask={isForMask}
              onDragStateChange={onDragStateChange}
              variant="presenceBare"
            />
          </AdjustmentSubsection>

          <AdjustmentSubsection title={t('adjustments.effects.creative')}>
            <Slider
              label={t('adjustments.effects.glow')}
              max={100}
              min={0}
              onChange={(event) => handleAdjustmentChange(CreativeAdjustment.GlowAmount, event.target.value)}
              step={1}
              value={adjustments.glowAmount}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('adjustments.effects.halation')}
              max={100}
              min={0}
              onChange={(event) => handleAdjustmentChange(CreativeAdjustment.HalationAmount, event.target.value)}
              step={1}
              value={adjustments.halationAmount}
              onDragStateChange={onDragStateChange}
            />
            {!isForMask && (
              <>
                <Slider
                  label={t('adjustments.effects.lightFlares')}
                  max={100}
                  min={0}
                  onChange={(event) => handleAdjustmentChange(CreativeAdjustment.FlareAmount, event.target.value)}
                  step={1}
                  value={adjustments.flareAmount}
                  onDragStateChange={onDragStateChange}
                />
                <Slider
                  label={t('adjustments.details.centre')}
                  max={100}
                  min={-100}
                  onChange={(event) => handleAdjustmentChange(DetailsAdjustment.Centré, event.target.value)}
                  step={1}
                  value={adjustments.centré}
                  onDragStateChange={onDragStateChange}
                />
              </>
            )}
          </AdjustmentSubsection>

          {!isForMask && (
            <>
              <AdjustmentSubsection title={t('adjustments.effects.lut')}>
                <LUTControl
                  lutPath={adjustments.lutPath || null}
                  lutName={adjustments.lutName || null}
                  lutIntensity={adjustments.lutIntensity ?? 100}
                  onLutSelect={handleLutSelect}
                  onLutHover={onLutHover}
                  onIntensityChange={handleLutIntensityChange}
                  onClear={handleLutClear}
                  onDragStateChange={onDragStateChange}
                />
              </AdjustmentSubsection>

              {adjustmentVisibility.vignette !== false && (
                <AdjustmentSubsection title={t('adjustments.effects.vignette')}>
                  <Slider
                    label={t('adjustments.effects.amount')}
                    max={100}
                    min={-100}
                    onChange={(event) => handleAdjustmentChange(Effect.VignetteAmount, event.target.value)}
                    step={1}
                    value={adjustments.vignetteAmount}
                    onDragStateChange={onDragStateChange}
                  />
                  <Slider
                    defaultValue={50}
                    label={t('adjustments.effects.midpoint')}
                    max={100}
                    min={0}
                    onChange={(event) => handleAdjustmentChange(Effect.VignetteMidpoint, event.target.value)}
                    step={1}
                    value={adjustments.vignetteMidpoint}
                    onDragStateChange={onDragStateChange}
                    fillOrigin="min"
                  />
                  <Slider
                    label={t('adjustments.effects.roundness')}
                    max={100}
                    min={-100}
                    onChange={(event) => handleAdjustmentChange(Effect.VignetteRoundness, event.target.value)}
                    step={1}
                    value={adjustments.vignetteRoundness}
                    onDragStateChange={onDragStateChange}
                  />
                  <Slider
                    defaultValue={50}
                    label={t('adjustments.effects.feather')}
                    max={100}
                    min={0}
                    onChange={(event) => handleAdjustmentChange(Effect.VignetteFeather, event.target.value)}
                    step={1}
                    value={adjustments.vignetteFeather}
                    onDragStateChange={onDragStateChange}
                    fillOrigin="min"
                  />
                </AdjustmentSubsection>
              )}

              {adjustmentVisibility.grain !== false && (
                <AdjustmentSubsection title={t('adjustments.effects.grain')}>
                  <Slider
                    label={t('adjustments.effects.amount')}
                    max={100}
                    min={0}
                    onChange={(event) => handleAdjustmentChange(Effect.GrainAmount, event.target.value)}
                    step={1}
                    value={adjustments.grainAmount}
                    onDragStateChange={onDragStateChange}
                  />
                  <Slider
                    defaultValue={25}
                    label={t('adjustments.effects.size')}
                    max={100}
                    min={0}
                    onChange={(event) => handleAdjustmentChange(Effect.GrainSize, event.target.value)}
                    step={1}
                    value={adjustments.grainSize}
                    onDragStateChange={onDragStateChange}
                    fillOrigin="min"
                  />
                  <Slider
                    defaultValue={50}
                    label={t('adjustments.effects.roughness')}
                    max={100}
                    min={0}
                    onChange={(event) => handleAdjustmentChange(Effect.GrainRoughness, event.target.value)}
                    step={1}
                    value={adjustments.grainRoughness}
                    onDragStateChange={onDragStateChange}
                    fillOrigin="min"
                  />
                </AdjustmentSubsection>
              )}
            </>
          )}
        </>
      )}

      {showLensBlur && !isForMask && (
        <AdjustmentSubsection>
          <Switch
            label={t('adjustments.effects.lensBlur')}
            checked={!!adjustments.lensBlurEnabled}
            onChange={handleLensBlurToggle}
          />

          {adjustments.lensBlurEnabled && (
            <div className="space-y-4 pt-3 pb-1">
              {isGeneratingDepth ? (
                <div className="camera-raw-status" role="status">
                  <Loader2 aria-hidden="true" size={16} className="animate-spin shrink-0" />
                  <Text variant={TextVariants.label}>
                    {aiModelDownloadStatus
                      ? t('editor.masks.settings.aiModelDownloading')
                      : t('editor.ai.generatingDepthMap')}
                  </Text>
                  {aiModelDownloadStatus && (
                    <Text variant={TextVariants.small} className="text-accent">
                      {aiModelDownloadStatus}
                    </Text>
                  )}
                </div>
              ) : (
                <>
                  <Slider
                    label={t('adjustments.effects.amount')}
                    max={100}
                    min={0}
                    defaultValue={40}
                    onChange={(event) => handleAdjustmentChange(Effect.LensBlurAmount, event.target.value)}
                    step={1}
                    value={adjustments.lensBlurAmount ?? 40}
                    onDragStateChange={onDragStateChange}
                    fillOrigin="min"
                  />
                  <Slider
                    label={t('adjustments.effects.lensDiffusion')}
                    max={100}
                    min={0}
                    defaultValue={0}
                    onChange={(event) => handleAdjustmentChange(Effect.lensBlurDiffusion, event.target.value)}
                    step={1}
                    value={adjustments.lensBlurDiffusion ?? 0}
                    onDragStateChange={onDragStateChange}
                  />
                  <BokehShapeSwitch
                    selectedShape={adjustments.lensBlurShape || 'circle'}
                    onShapeChange={(shapeId) =>
                      setAdjustments((prev: Partial<Adjustments>) => ({
                        ...prev,
                        [Effect.LensBlurShape]: shapeId,
                      }))
                    }
                  />
                  <DepthRangePicker
                    minDepth={100 - (adjustments.lensBlurMaxDepth ?? 100)}
                    maxDepth={100 - (adjustments.lensBlurMinDepth ?? 20)}
                    minFade={adjustments.lensBlurMaxFade ?? 20}
                    maxFade={adjustments.lensBlurMinFade ?? 20}
                    defaultMinDepth={0}
                    defaultMaxDepth={80}
                    defaultMinFade={20}
                    defaultMaxFade={20}
                    onChange={(values: { minDepth: number; maxDepth: number; minFade: number; maxFade: number }) => {
                      setAdjustments((prev: Partial<Adjustments>) => ({
                        ...prev,
                        lensBlurMinDepth: 100 - values.maxDepth,
                        lensBlurMaxDepth: 100 - values.minDepth,
                        lensBlurMinFade: values.maxFade,
                        lensBlurMaxFade: values.minFade,
                      }));
                    }}
                    onDragStateChange={onDragStateChange}
                  />
                </>
              )}
            </div>
          )}
        </AdjustmentSubsection>
      )}
    </div>
  );
}
