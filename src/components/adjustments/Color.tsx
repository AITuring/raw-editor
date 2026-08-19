import { useCallback, useId, useMemo, useRef, useState, type CSSProperties } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, Pipette, Sliders } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { message } from '../ui/messageApi';
import Slider from '../ui/Slider';
import ColorWheel from '../ui/ColorWheel';
import { ColorAdjustment, ColorCalibration, HueSatLum, INITIAL_ADJUSTMENTS } from '../../utils/adjustments';
import { Adjustments, ColorGrading } from '../../utils/adjustments';
import { AppSettings, Invokes, SelectedImage } from '../ui/AppProperties';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import { AdjustmentSubsection, AdjustmentTabs } from '../ui/AdjustmentSubsection';
import { useEditorStore } from '../../store/useEditorStore';
import {
  WHITE_BALANCE_PRESETS,
  cameraRawTintToRelative,
  inferAsShotKelvin,
  kelvinToRelativeTemperature,
  relativeTemperatureToKelvin,
  relativeTintToCameraRaw,
  type WhiteBalanceMode,
} from '../../utils/whiteBalance';

interface ColorProps {
  baseHue: number;
  name: string;
  label: string;
}

interface ColorPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((previous: Adjustments) => Adjustments)): any;
  appSettings: AppSettings | null;
  selectedImage?: SelectedImage | null;
  isForMask?: boolean;
  isWbPickerActive?: boolean;
  toggleWbPicker?: () => void;
  onDragStateChange?: (isDragging: boolean) => void;
  variant?: 'all' | 'whiteBalance' | 'presence' | 'presenceBare' | 'mixer' | 'grading' | 'calibration';
}

const ColorGradingPanel = ({ adjustments, setAdjustments, onDragStateChange }: ColorPanelProps) => {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState<'3way' | 'global'>('3way');
  const [isExpanded, setIsExpanded] = useState(false);
  const colorGrading = adjustments.colorGrading || INITIAL_ADJUSTMENTS.colorGrading;

  const handleChange = (grading: ColorGrading, newValue: HueSatLum) => {
    setAdjustments((prev: Partial<Adjustments>) => ({
      ...prev,
      colorGrading: {
        ...(prev.colorGrading || INITIAL_ADJUSTMENTS.colorGrading),
        [grading]: newValue,
      },
    }));
  };

  const handleColorGradingSliderChange = (grading: ColorGrading, value: string) => {
    setAdjustments((prev: Partial<Adjustments>) => ({
      ...prev,
      colorGrading: {
        ...(prev.colorGrading || INITIAL_ADJUSTMENTS.colorGrading),
        [grading]: parseFloat(value),
      },
    }));
  };

  const tabs = useMemo(
    () => [
      {
        id: '3way',
        label: t('adjustments.color.grading.threeWay'),
        icon: (
          <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="6" r="4.5" />
            <circle cx="5" cy="18" r="4.5" />
            <circle cx="19" cy="18" r="4.5" />
          </svg>
        ),
      },
      {
        id: 'global',
        label: t('adjustments.color.grading.global'),
        icon: (
          <div
            aria-hidden="true"
            className="w-3.5 h-3.5 rounded-full"
            style={{ background: 'linear-gradient(to top, #666, #fff)' }}
          />
        ),
      },
    ],
    [],
  );

  return (
    <div>
      <div className="flex items-center justify-start gap-2 mb-4 mt-2">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              aria-label={tab.label}
              aria-pressed={isActive}
              key={tab.id}
              data-tooltip={tab.label}
              onClick={() => setActiveTab(tab.id as '3way' | 'global')}
              className={`w-7 h-7 rounded-full flex items-center justify-center transition-colors
                ${
                  isActive
                    ? 'ring-2 ring-offset-2 ring-offset-surface ring-accent text-text-primary'
                    : 'bg-bg-secondary text-text-secondary hover:text-text-primary hover:bg-bg-secondary/80'
                }`}
              type="button"
            >
              {tab.icon}
            </button>
          );
        })}

        <div className="w-px h-5 bg-text-secondary/20 mx-1" />

        <button
          aria-label={t('adjustments.color.toggleSliders')}
          aria-pressed={isExpanded}
          onClick={() => setIsExpanded(!isExpanded)}
          className={`w-7 h-7 rounded-full flex items-center justify-center transition-colors
            ${
              isExpanded
                ? 'bg-accent text-button-text'
                : 'bg-bg-secondary text-text-secondary hover:text-text-primary hover:bg-bg-secondary/80'
            }`}
          data-tooltip={t('adjustments.color.toggleSliders')}
          type="button"
        >
          <Sliders aria-hidden="true" size={14} />
        </button>
      </div>

      <div className="relative w-full mb-4">
        <AnimatePresence mode="wait">
          {activeTab === '3way' ? (
            <motion.div
              key="3way"
              initial={{ opacity: 0, x: -15 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -15 }}
              transition={{ duration: 0.2 }}
              className="w-full"
            >
              <div className="flex justify-center mb-4">
                <div className="w-[calc(50%-0.5rem)]">
                  <ColorWheel
                    defaultValue={INITIAL_ADJUSTMENTS.colorGrading.midtones}
                    label={t('adjustments.color.grading.midtones')}
                    onChange={(val: HueSatLum) => handleChange(ColorGrading.Midtones, val)}
                    value={colorGrading.midtones}
                    onDragStateChange={onDragStateChange}
                    isExpanded={isExpanded}
                  />
                </div>
              </div>
              <div className="flex justify-between mb-2 gap-4">
                <div className="w-full flex-1 min-w-0">
                  <ColorWheel
                    defaultValue={INITIAL_ADJUSTMENTS.colorGrading.shadows}
                    label={t('adjustments.color.grading.shadows')}
                    onChange={(val: HueSatLum) => handleChange(ColorGrading.Shadows, val)}
                    value={colorGrading.shadows}
                    onDragStateChange={onDragStateChange}
                    isExpanded={isExpanded}
                  />
                </div>
                <div className="w-full flex-1 min-w-0">
                  <ColorWheel
                    defaultValue={INITIAL_ADJUSTMENTS.colorGrading.highlights}
                    label={t('adjustments.color.grading.highlights')}
                    onChange={(val: HueSatLum) => handleChange(ColorGrading.Highlights, val)}
                    value={colorGrading.highlights}
                    onDragStateChange={onDragStateChange}
                    isExpanded={isExpanded}
                  />
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="global"
              initial={{ opacity: 0, x: 15 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 15 }}
              transition={{ duration: 0.2 }}
              className="w-full flex justify-center pb-2"
            >
              <div className="w-full max-w-70">
                <ColorWheel
                  defaultValue={INITIAL_ADJUSTMENTS.colorGrading.global}
                  label={t('adjustments.color.grading.global')}
                  onChange={(val: HueSatLum) => handleChange(ColorGrading.Global, val)}
                  value={colorGrading.global || INITIAL_ADJUSTMENTS.colorGrading.global}
                  onDragStateChange={onDragStateChange}
                  isExpanded={isExpanded}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div>
        <Slider
          defaultValue={50}
          label={t('adjustments.color.grading.blending')}
          max={100}
          min={0}
          onChange={(e: any) => handleColorGradingSliderChange(ColorGrading.Blending, e.target.value)}
          step={1}
          value={colorGrading.blending}
          onDragStateChange={onDragStateChange}
        />
        <Slider
          defaultValue={0}
          label={t('adjustments.color.grading.balance')}
          max={100}
          min={-100}
          onChange={(e: any) => handleColorGradingSliderChange(ColorGrading.Balance, e.target.value)}
          step={1}
          value={colorGrading.balance}
          onDragStateChange={onDragStateChange}
        />
      </div>
    </div>
  );
};

const ColorCalibrationPanel = ({ adjustments, setAdjustments, onDragStateChange }: ColorPanelProps) => {
  const { t } = useTranslation();
  const colorCalibration = adjustments.colorCalibration || INITIAL_ADJUSTMENTS.colorCalibration;

  const primaryColors = useMemo(
    () =>
      [
        { name: 'red' as const, label: t('adjustments.color.calibration.colors.red') },
        { name: 'green' as const, label: t('adjustments.color.calibration.colors.green') },
        { name: 'blue' as const, label: t('adjustments.color.calibration.colors.blue') },
      ] as const,
    [t],
  );

  const handleShadowsChange = (value: string) => {
    setAdjustments((prev: Partial<Adjustments>) => ({
      ...prev,
      colorCalibration: {
        ...(prev.colorCalibration || INITIAL_ADJUSTMENTS.colorCalibration),
        shadowsTint: parseFloat(value),
      },
    }));
  };

  const handlePrimaryChange = (primary: 'red' | 'green' | 'blue', key: 'Hue' | 'Saturation', value: string) => {
    const fullKey = `${primary}${key}` as keyof ColorCalibration;
    setAdjustments((prev: Partial<Adjustments>) => ({
      ...prev,
      colorCalibration: {
        ...(prev.colorCalibration || INITIAL_ADJUSTMENTS.colorCalibration),
        [fullKey]: parseFloat(value),
      },
    }));
  };

  return (
    <div className="camera-raw-section-body">
      <AdjustmentSubsection title={t('adjustments.color.calibration.shadows')}>
        <Slider
          label={t('adjustments.color.calibration.tint')}
          min={-100}
          max={100}
          step={1}
          defaultValue={0}
          value={colorCalibration.shadowsTint}
          onChange={(e: any) => handleShadowsChange(e.target.value)}
          onDragStateChange={onDragStateChange}
          trackClassName="tint-gradient-track"
        />
      </AdjustmentSubsection>

      {primaryColors.map(({ name, label }) => (
        <AdjustmentSubsection key={name} title={label}>
          <Slider
            label={t('adjustments.color.calibration.hue')}
            min={-100}
            max={100}
            step={1}
            defaultValue={0}
            value={colorCalibration[`${name}Hue` as keyof ColorCalibration]}
            onChange={(event) => handlePrimaryChange(name, 'Hue', String(event.target.value))}
            onDragStateChange={onDragStateChange}
            trackClassName={`hue-slider-${name}s`}
          />
          <Slider
            label={t('adjustments.color.calibration.saturation')}
            min={-100}
            max={100}
            step={1}
            defaultValue={0}
            value={colorCalibration[`${name}Saturation` as keyof ColorCalibration]}
            onChange={(event) => handlePrimaryChange(name, 'Saturation', String(event.target.value))}
            onDragStateChange={onDragStateChange}
            trackClassName={`sat-slider-${name}s`}
          />
        </AdjustmentSubsection>
      ))}
    </div>
  );
};

export default function ColorPanel({
  adjustments,
  setAdjustments,
  appSettings,
  selectedImage,
  isForMask = false,
  isWbPickerActive = false,
  toggleWbPicker,
  onDragStateChange,
  variant = 'all',
}: ColorPanelProps) {
  const { t } = useTranslation();
  const [mixerChannel, setMixerChannel] = useState<'hue' | 'saturation' | 'luminance' | 'all'>('hue');
  const [whiteBalanceRequestPath, setWhiteBalanceRequestPath] = useState<string | null>(null);
  const whiteBalanceRequestId = useRef(0);
  const whiteBalanceModeId = useId();
  const adjustmentVisibility = appSettings?.adjustmentVisibility || {};
  const HSL_COLORS = useMemo<Array<ColorProps>>(
    () => [
      { name: 'reds', baseHue: 0, label: t('adjustments.color.mixerColors.reds') },
      { name: 'oranges', baseHue: 30, label: t('adjustments.color.mixerColors.oranges') },
      { name: 'yellows', baseHue: 60, label: t('adjustments.color.mixerColors.yellows') },
      { name: 'greens', baseHue: 120, label: t('adjustments.color.mixerColors.greens') },
      { name: 'aquas', baseHue: 180, label: t('adjustments.color.mixerColors.aquas') },
      { name: 'blues', baseHue: 240, label: t('adjustments.color.mixerColors.blues') },
      { name: 'purples', baseHue: 300, label: t('adjustments.color.mixerColors.purples') },
      { name: 'magentas', baseHue: 340, label: t('adjustments.color.mixerColors.magentas') },
    ],
    [t],
  );

  const handleAdjustmentChange = (key: ColorAdjustment, value: string) => {
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      [key]: parseFloat(value),
      ...(key === ColorAdjustment.Temperature || key === ColorAdjustment.Tint
        ? { whiteBalanceMode: 'custom' as const }
        : {}),
    }));
  };

  const asShotKelvin = useMemo(() => inferAsShotKelvin(selectedImage?.exif), [selectedImage?.exif]);
  const temperatureToKelvin = useCallback(
    (temperature: number) => relativeTemperatureToKelvin(temperature, asShotKelvin),
    [asShotKelvin],
  );
  const kelvinToTemperature = useCallback(
    (kelvin: number) => kelvinToRelativeTemperature(kelvin, asShotKelvin),
    [asShotKelvin],
  );
  const whiteBalanceMode = (adjustments.whiteBalanceMode ||
    (adjustments.temperature !== 0 || adjustments.tint !== 0 ? 'custom' : 'asShot')) as WhiteBalanceMode;
  const isCalculatingWhiteBalance = whiteBalanceRequestPath !== null && whiteBalanceRequestPath === selectedImage?.path;
  const whiteBalanceOptions = useMemo(
    () => [
      { value: 'asShot' as const, label: t('adjustments.color.whiteBalanceModes.asShot') },
      { value: 'auto' as const, label: t('adjustments.color.whiteBalanceModes.auto') },
      { value: 'daylight' as const, label: t('adjustments.color.whiteBalanceModes.daylight') },
      { value: 'cloudy' as const, label: t('adjustments.color.whiteBalanceModes.cloudy') },
      { value: 'shade' as const, label: t('adjustments.color.whiteBalanceModes.shade') },
      { value: 'tungsten' as const, label: t('adjustments.color.whiteBalanceModes.tungsten') },
      { value: 'fluorescent' as const, label: t('adjustments.color.whiteBalanceModes.fluorescent') },
      { value: 'flash' as const, label: t('adjustments.color.whiteBalanceModes.flash') },
      { value: 'custom' as const, label: t('adjustments.color.whiteBalanceModes.custom') },
    ],
    [t],
  );

  const handleWhiteBalanceModeChange = async (mode: WhiteBalanceMode) => {
    if (mode === 'asShot') {
      setAdjustments((previous: Adjustments) => ({
        ...previous,
        temperature: 0,
        tint: 0,
        whiteBalanceMode: 'asShot',
      }));
      return;
    }

    if (mode === 'custom') {
      setAdjustments((previous: Adjustments) => ({ ...previous, whiteBalanceMode: 'custom' }));
      return;
    }

    if (mode === 'auto') {
      const requestedPath = selectedImage?.path;
      if (!requestedPath || !selectedImage?.isReady) return;

      const previousMode = whiteBalanceMode;
      const requestId = ++whiteBalanceRequestId.current;
      setWhiteBalanceRequestPath(requestedPath);
      setAdjustments((previous: Adjustments) => ({ ...previous, whiteBalanceMode: 'auto' }));

      try {
        const automatic = await invoke<Pick<Adjustments, 'temperature' | 'tint'>>(Invokes.CalculateAutoAdjustments);
        const activeImagePath = useEditorStore.getState().selectedImage?.path;
        if (whiteBalanceRequestId.current !== requestId || activeImagePath !== requestedPath) return;

        setAdjustments((previous: Adjustments) => ({
          ...previous,
          temperature: Number(automatic.temperature) || 0,
          tint: Number(automatic.tint) || 0,
          whiteBalanceMode: 'auto',
        }));
      } catch (error) {
        const activeImagePath = useEditorStore.getState().selectedImage?.path;
        if (whiteBalanceRequestId.current === requestId && activeImagePath === requestedPath) {
          setAdjustments((previous: Adjustments) => ({ ...previous, whiteBalanceMode: previousMode }));
          message.error(t('adjustments.color.autoWhiteBalanceFailed', { error: String(error) }));
        }
      } finally {
        if (whiteBalanceRequestId.current === requestId) setWhiteBalanceRequestPath(null);
      }
      return;
    }

    const preset = WHITE_BALANCE_PRESETS[mode];
    setAdjustments((previous: Adjustments) => ({
      ...previous,
      temperature: kelvinToRelativeTemperature(preset.kelvin, asShotKelvin),
      tint: cameraRawTintToRelative(preset.tint),
      whiteBalanceMode: mode,
    }));
  };

  const handleHslChange = (colorName: string, key: 'hue' | 'saturation' | 'luminance', value: string) => {
    setAdjustments((prev: Partial<Adjustments>) => ({
      ...prev,
      hsl: {
        ...(prev.hsl || {}),
        [colorName]: {
          ...(prev.hsl?.[colorName] || INITIAL_ADJUSTMENTS.hsl[colorName]),
          [key]: parseFloat(value),
        },
      },
    }));
  };

  const mixerTabs = useMemo(
    () => [
      { id: 'hue' as const, label: t('adjustments.color.hue') },
      { id: 'saturation' as const, label: t('adjustments.color.saturation') },
      { id: 'luminance' as const, label: t('adjustments.color.luminance') },
      { id: 'all' as const, label: t('adjustments.color.all') },
    ],
    [t],
  );
  const visibleMixerChannels =
    mixerChannel === 'all' ? (['hue', 'saturation', 'luminance'] as const) : ([mixerChannel] as const);
  const showWhiteBalance = variant === 'all' || variant === 'whiteBalance';
  const showPresence = variant === 'all' || variant === 'presence' || variant === 'presenceBare';
  const showMixer = variant === 'all' || variant === 'mixer';
  const showGrading = variant === 'all' || variant === 'grading';
  const showCalibration = variant === 'all' || variant === 'calibration';

  return (
    <div className="space-y-4">
      {showWhiteBalance && (
        <div className="camera-raw-white-balance">
          <div className="camera-raw-field camera-raw-white-balance-field">
            <label className="camera-raw-field-label" htmlFor={whiteBalanceModeId}>
              {t('adjustments.color.whiteBalance')}
            </label>
            <div className="camera-raw-white-balance-control">
              <select
                aria-label={t('adjustments.color.whiteBalance')}
                className="camera-raw-select"
                disabled={isCalculatingWhiteBalance}
                id={whiteBalanceModeId}
                onChange={(event) => void handleWhiteBalanceModeChange(event.target.value as WhiteBalanceMode)}
                value={whiteBalanceMode}
              >
                {whiteBalanceOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {!isForMask && toggleWbPicker && (
                <button
                  aria-label={t('adjustments.color.wbPickerTooltip')}
                  aria-pressed={isWbPickerActive}
                  onClick={toggleWbPicker}
                  className={`camera-raw-wb-picker ${isWbPickerActive ? 'is-active' : ''}`}
                  data-tooltip={t('adjustments.color.wbPickerTooltip')}
                  type="button"
                >
                  <Pipette aria-hidden="true" size={16} />
                </button>
              )}
            </div>
          </div>
          {isCalculatingWhiteBalance && (
            <div
              aria-live="polite"
              className="camera-raw-white-balance-status semantic-status"
              data-tone="processing"
              role="status"
            >
              <Loader2 aria-hidden="true" className="animate-spin" size={11} />
              <span>{t('adjustments.color.calculatingWhiteBalance')}</span>
            </div>
          )}
          <Slider
            disabled={isCalculatingWhiteBalance}
            displayDecimals={0}
            displayStep={50}
            fromDisplayValue={kelvinToTemperature}
            label={t('adjustments.color.temperature')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(ColorAdjustment.Temperature, e.target.value)}
            showPositiveSign={false}
            step={1}
            suffix="K"
            toDisplayValue={temperatureToKelvin}
            value={adjustments.temperature || 0}
            trackClassName="temperature-gradient-track"
            onDragStateChange={onDragStateChange}
          />
          <Slider
            disabled={isCalculatingWhiteBalance}
            displayDecimals={0}
            displayStep={1}
            fromDisplayValue={cameraRawTintToRelative}
            label={t('adjustments.color.tint')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(ColorAdjustment.Tint, e.target.value)}
            step={1}
            toDisplayValue={relativeTintToCameraRaw}
            value={adjustments.tint || 0}
            trackClassName="tint-gradient-track"
            onDragStateChange={onDragStateChange}
          />
        </div>
      )}

      {showPresence && (
        <div className={variant === 'presenceBare' ? '' : 'p-2 bg-bg-tertiary rounded-md'}>
          {variant !== 'presenceBare' && (
            <Text variant={TextVariants.heading} className="mb-2">
              {t('adjustments.color.presence')}
            </Text>
          )}
          <Slider
            label={t('adjustments.color.vibrance')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(ColorAdjustment.Vibrance, e.target.value)}
            step={1}
            value={adjustments.vibrance || 0}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.color.saturation')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(ColorAdjustment.Saturation, e.target.value)}
            step={1}
            value={adjustments.saturation || 0}
            onDragStateChange={onDragStateChange}
          />
        </div>
      )}

      {showMixer && (
        <div className="camera-raw-section-body">
          <AdjustmentSubsection title={isForMask ? t('adjustments.color.localHue') : t('adjustments.color.hue')}>
            <Slider
              label={t('adjustments.color.hue')}
              max={180}
              min={-180}
              onChange={(e: any) => handleAdjustmentChange(ColorAdjustment.Hue, e.target.value)}
              step={1}
              value={adjustments.hue || 0}
              trackClassName="hue-range-track"
              onDragStateChange={onDragStateChange}
            />
          </AdjustmentSubsection>

          <AdjustmentSubsection>
            <AdjustmentTabs
              ariaLabel={t('adjustments.color.colorMixer')}
              onChange={setMixerChannel}
              tabs={mixerTabs}
              value={mixerChannel}
            />

            <div className="camera-raw-mixer-channels">
              {visibleMixerChannels.map((channel) => (
                <div className="camera-raw-mixer-channel" key={channel}>
                  {mixerChannel === 'all' && (
                    <div className="camera-raw-mixer-channel-title">
                      {mixerTabs.find((tab) => tab.id === channel)?.label}
                    </div>
                  )}
                  {HSL_COLORS.map(({ baseHue, name, label }) => {
                    const colorValues = adjustments.hsl?.[name] || INITIAL_ADJUSTMENTS.hsl[name];
                    const effectiveHue = (((baseHue + colorValues.hue) % 360) + 360) % 360;
                    const effectiveSaturation = (colorValues.saturation + 100) / 2;
                    const trackPrefix = channel === 'luminance' ? 'lum' : channel === 'saturation' ? 'sat' : 'hue';
                    const customProperties = {
                      [`--hsl-mixer-hue-${name}`]: effectiveHue,
                      [`--hsl-mixer-sat-${name}`]: `${effectiveSaturation}%`,
                    } as CSSProperties;

                    return (
                      <div key={`${channel}-${name}`} style={customProperties}>
                        <Slider
                          label={label}
                          max={100}
                          min={-100}
                          onChange={(event) => handleHslChange(name, channel, String(event.target.value))}
                          step={1}
                          value={colorValues[channel]}
                          trackClassName={`${trackPrefix}-slider-${name}`}
                          onDragStateChange={onDragStateChange}
                        />
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </AdjustmentSubsection>
        </div>
      )}

      {showGrading && (
        <div className={variant === 'grading' ? '' : 'p-2 bg-bg-tertiary rounded-md'}>
          {variant === 'all' && (
            <Text variant={TextVariants.heading} className="mb-3">
              {t('adjustments.color.colorGrading')}
            </Text>
          )}
          <ColorGradingPanel
            adjustments={adjustments}
            setAdjustments={setAdjustments}
            appSettings={appSettings}
            onDragStateChange={onDragStateChange}
          />
        </div>
      )}

      {!isForMask && showCalibration && adjustmentVisibility.colorCalibration !== false && (
        <ColorCalibrationPanel
          adjustments={adjustments}
          setAdjustments={setAdjustments}
          appSettings={appSettings}
          onDragStateChange={onDragStateChange}
          variant={variant}
        />
      )}
    </div>
  );
}
