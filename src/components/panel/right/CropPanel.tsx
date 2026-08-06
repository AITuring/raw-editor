import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Aperture,
  FlipHorizontal,
  FlipVertical,
  Lock,
  LockOpen,
  RectangleHorizontal,
  RectangleVertical,
  RotateCcw,
  RotateCw,
  Ruler,
  Scan,
  X,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Adjustments, INITIAL_ADJUSTMENTS } from '../../../utils/adjustments';
import clsx from 'clsx';
import { Orientation } from '../../ui/AppProperties';
import TransformModal from '../../modals/TransformModal';
import LensCorrectionModal from '../../modals/LensCorrectionModal';
import Text from '../../ui/Text';
import Slider from '../../ui/Slider';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useEditorStore } from '../../../store/useEditorStore';
import { useEditorActions } from '../../../hooks/useEditorActions';
import { calculateAreaPreservingCrop, calculateCenteredCrop } from '../../../utils/cropUtils';
import { Crop } from 'react-image-crop';

const BASE_RATIO = 1.618;
const ORIGINAL_RATIO = 0;
const RATIO_TOLERANCE = 0.01;

export type OverlayMode = 'none' | 'thirds' | 'goldenTriangle' | 'goldenSpiral' | 'phiGrid' | 'armature' | 'diagonal';

interface CropPreset {
  id: string;
  name: string;
  value: number | null;
  tooltip: string;
}

interface OverlayOption {
  id: OverlayMode;
  name: string;
  tooltip: string;
}

export default function CropPanel() {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((s) => s.selectedImage);
  const adjustments = useEditorStore((s) => s.adjustments);
  const isStraightenActive = useEditorStore((s) => s.isStraightenActive);
  const activeOverlay = useEditorStore((s) => s.overlayMode);
  const setEditor = useEditorStore((s) => s.setEditor);
  const { setAdjustments } = useEditorActions();
  const [customW, setCustomW] = useState('');
  const [customH, setCustomH] = useState('');
  const [isTransformModalOpen, setIsTransformModalOpen] = useState(false);
  const [isLensModalOpen, setIsLensModalOpen] = useState(false);
  const [isRotationActive, setIsRotationActive] = useState(false);
  const [preferPortrait, setPreferPortrait] = useState(false);
  const [isEditingCustom, setIsEditingCustom] = useState(false);
  const [displayPresetId, setDisplayPresetId] = useState('free');

  const [localRotation, setLocalRotation] = useState<number | null>(null);
  const localRotationRef = useRef<number | null>(null);
  const lastConstrainedRatioRef = useRef<number | null>(null);
  const preferredPresetIdRef = useRef<string | null>(null);
  const selectedImagePathRef = useRef<string | null>(null);

  const PRESETS = useMemo<Array<CropPreset>>(
    () => [
      {
        id: 'free',
        name: t('editor.crop.presets.free.name'),
        value: null,
        tooltip: t('editor.crop.presets.free.desc'),
      },
      {
        id: 'original',
        name: t('editor.crop.presets.original.name'),
        value: ORIGINAL_RATIO,
        tooltip: t('editor.crop.presets.original.desc'),
      },
      { id: 'square', name: t('editor.crop.presets.sq.name'), value: 1, tooltip: t('editor.crop.presets.sq.desc') },
      { id: '5x4', name: t('editor.crop.presets.r54.name'), value: 5 / 4, tooltip: t('editor.crop.presets.r54.desc') },
      { id: '4x3', name: t('editor.crop.presets.r43.name'), value: 4 / 3, tooltip: t('editor.crop.presets.r43.desc') },
      { id: '3x2', name: t('editor.crop.presets.r32.name'), value: 3 / 2, tooltip: t('editor.crop.presets.r32.desc') },
      { id: '5x7', name: '5:7', value: 7 / 5, tooltip: '5:7' },
      {
        id: '16x9',
        name: t('editor.crop.presets.r169.name'),
        value: 16 / 9,
        tooltip: t('editor.crop.presets.r169.desc'),
      },
      {
        id: '21x9',
        name: t('editor.crop.presets.r219.name'),
        value: 21 / 9,
        tooltip: t('editor.crop.presets.r219.desc'),
      },
      {
        id: '65x24',
        name: t('editor.crop.presets.r6524.name'),
        value: 65 / 24,
        tooltip: t('editor.crop.presets.r6524.desc'),
      },
    ],
    [t],
  );

  const OVERLAYS = useMemo<Array<OverlayOption>>(
    () => [
      { id: 'none', name: t('editor.crop.overlays.none.name'), tooltip: t('editor.crop.overlays.none.desc') },
      { id: 'thirds', name: t('editor.crop.overlays.thirds.name'), tooltip: t('editor.crop.overlays.thirds.desc') },
      {
        id: 'diagonal',
        name: t('editor.crop.overlays.diagonal.name'),
        tooltip: t('editor.crop.overlays.diagonal.desc'),
      },
      {
        id: 'goldenTriangle',
        name: t('editor.crop.overlays.triangle.name'),
        tooltip: t('editor.crop.overlays.triangle.desc'),
      },
      {
        id: 'goldenSpiral',
        name: t('editor.crop.overlays.spiral.name'),
        tooltip: t('editor.crop.overlays.spiral.desc'),
      },
      { id: 'phiGrid', name: t('editor.crop.overlays.phiGrid.name'), tooltip: t('editor.crop.overlays.phiGrid.desc') },
      {
        id: 'armature',
        name: t('editor.crop.overlays.armature.name'),
        tooltip: t('editor.crop.overlays.armature.desc'),
      },
    ],
    [t],
  );

  const updateLocalRotation = useCallback(
    (val: number | null) => {
      setLocalRotation(val);
      localRotationRef.current = val;
      setEditor({ liveRotation: val });
    },
    [setEditor],
  );

  const setOverlay = useCallback((mode: OverlayMode) => setEditor({ overlayMode: mode }), [setEditor]);

  const setOverlayRotation = useCallback(
    (updater: React.SetStateAction<number>) => {
      setEditor((state) => ({
        overlayRotation: typeof updater === 'function' ? updater(state.overlayRotation) : updater,
      }));
    },
    [setEditor],
  );

  const lastSyncedRatio = useRef<number | null>(null);

  const { aspectRatio, rotation = 0, flipHorizontal = false, flipVertical = false, orientationSteps = 0 } = adjustments;

  useEffect(() => {
    if (isStraightenActive) {
      updateLocalRotation(null);
    }
  }, [isStraightenActive, updateLocalRotation]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeTag = document.activeElement?.tagName.toLowerCase();
      if (activeTag === 'input' || activeTag === 'textarea') return;

      if (e.ctrlKey || e.metaKey) return;

      if (e.key.toLowerCase() === 'o') {
        e.preventDefault();

        if (e.shiftKey) {
          setOverlayRotation((prev) => (prev + 1) % 4);
        } else {
          const currentIndex = OVERLAYS.findIndex((o) => o.id === activeOverlay);
          const nextIndex = (currentIndex + 1) % OVERLAYS.length;
          setOverlay(OVERLAYS[nextIndex].id);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeOverlay, setOverlay, setOverlayRotation, OVERLAYS]);

  useEffect(() => {
    return () => {
      setEditor({ liveRotation: null });
    };
  }, [setEditor]);

  const getEffectiveOriginalRatio = useCallback(() => {
    if (!selectedImage?.width || !selectedImage?.height) {
      return null;
    }
    const isSwapped = orientationSteps === 1 || orientationSteps === 3;
    const W = isSwapped ? selectedImage.height : selectedImage.width;
    const H = isSwapped ? selectedImage.width : selectedImage.height;
    return W > 0 && H > 0 ? W / H : null;
  }, [selectedImage, orientationSteps]);

  const activePreset = useMemo(() => {
    if (aspectRatio === null) {
      return PRESETS.find((p: CropPreset) => p.value === null);
    }

    const originalRatio = getEffectiveOriginalRatio();
    if (originalRatio && Math.abs(aspectRatio - originalRatio) < RATIO_TOLERANCE) {
      return PRESETS.find((p: CropPreset) => p.value === ORIGINAL_RATIO);
    }

    const numericPresetMatch = PRESETS.find(
      (p: CropPreset) =>
        p.value &&
        p.value !== ORIGINAL_RATIO &&
        (Math.abs(aspectRatio - p.value) < RATIO_TOLERANCE || Math.abs(aspectRatio - 1 / p.value) < RATIO_TOLERANCE),
    );

    if (numericPresetMatch) {
      return numericPresetMatch;
    }

    return null;
  }, [aspectRatio, getEffectiveOriginalRatio, PRESETS]);

  useEffect(() => {
    const imagePath = selectedImage?.path ?? null;
    const imageChanged = selectedImagePathRef.current !== imagePath;
    selectedImagePathRef.current = imagePath;

    if (aspectRatio === null) {
      if (imageChanged) {
        setDisplayPresetId('free');
        lastConstrainedRatioRef.current = null;
      }
      preferredPresetIdRef.current = null;
      return;
    }

    lastConstrainedRatioRef.current = aspectRatio;

    const presetMatchesRatio = (presetId: string) => {
      if (presetId === 'custom') return true;
      const preset = PRESETS.find((candidate) => candidate.id === presetId);
      if (!preset || preset.value === null) return false;
      if (preset.value === ORIGINAL_RATIO) {
        const originalRatio = getEffectiveOriginalRatio();
        return originalRatio !== null && Math.abs(aspectRatio - originalRatio) < RATIO_TOLERANCE;
      }
      return (
        Math.abs(aspectRatio - preset.value) < RATIO_TOLERANCE ||
        Math.abs(aspectRatio - 1 / preset.value) < RATIO_TOLERANCE
      );
    };

    const preferredPresetId = preferredPresetIdRef.current;
    if (preferredPresetId && presetMatchesRatio(preferredPresetId)) {
      setDisplayPresetId(preferredPresetId);
      preferredPresetIdRef.current = null;
      return;
    }
    preferredPresetIdRef.current = null;

    if (imageChanged || !presetMatchesRatio(displayPresetId)) {
      setDisplayPresetId(activePreset?.id ?? 'custom');
    }
  }, [activePreset, aspectRatio, displayPresetId, getEffectiveOriginalRatio, PRESETS, selectedImage?.path]);

  const effectiveAspectForUi = aspectRatio ?? lastConstrainedRatioRef.current;
  const orientation = effectiveAspectForUi && effectiveAspectForUi < 1 ? Orientation.Vertical : Orientation.Horizontal;
  const isAspectConstrained = aspectRatio !== null;
  const isCustomActive = displayPresetId === 'custom';

  useEffect(() => {
    if (aspectRatio && aspectRatio !== 1) {
      setPreferPortrait(aspectRatio < 1);
    }
  }, [aspectRatio]);

  useEffect(() => {
    const customRatio = aspectRatio ?? lastConstrainedRatioRef.current;
    if (isCustomActive && customRatio && !isEditingCustom) {
      if (lastSyncedRatio.current === null || Math.abs(lastSyncedRatio.current - customRatio) > RATIO_TOLERANCE) {
        const h = 100;
        const w = customRatio * h;
        setCustomW(w.toFixed(1).replace(/\.0$/, ''));
        setCustomH(h.toString());
        lastSyncedRatio.current = customRatio;
      }
    } else if (!isCustomActive) {
      lastSyncedRatio.current = null;
    }
  }, [isCustomActive, aspectRatio, isEditingCustom]);

  const applyAspectRatio = useCallback(
    (newAspectRatio: number | null) => {
      if (newAspectRatio === null) {
        setAdjustments((prev: Adjustments) => ({ ...prev, aspectRatio: null }));
        return;
      }
      let newCrop: Crop | null = null;
      if (selectedImage?.width && selectedImage?.height) {
        newCrop =
          calculateAreaPreservingCrop(
            selectedImage.width,
            selectedImage.height,
            orientationSteps,
            newAspectRatio,
            rotation,
            adjustments.crop,
          ) ??
          calculateCenteredCrop(selectedImage.width, selectedImage.height, orientationSteps, newAspectRatio, rotation);
      }
      setAdjustments((prev: Adjustments) => ({ ...prev, aspectRatio: newAspectRatio, crop: newCrop }));
    },
    [selectedImage, orientationSteps, rotation, adjustments.crop, setAdjustments],
  );

  useEffect(() => {
    if (displayPresetId === 'original') {
      const newOriginalRatio = getEffectiveOriginalRatio();
      if (newOriginalRatio !== null && aspectRatio && Math.abs(aspectRatio - newOriginalRatio) > RATIO_TOLERANCE) {
        preferredPresetIdRef.current = 'original';
        applyAspectRatio(newOriginalRatio);
      }
    }
  }, [orientationSteps, displayPresetId, aspectRatio, getEffectiveOriginalRatio, applyAspectRatio]);

  const handleCustomInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    if (name === 'customW') {
      setCustomW(value);
    } else if (name === 'customH') {
      setCustomH(value);
    }
  };

  const handleCustomInputFocus = () => {
    setIsEditingCustom(true);
  };

  const handleApplyCustomRatio = () => {
    setIsEditingCustom(false);
    const numW = parseFloat(customW);
    const numH = parseFloat(customH);

    if (numW > 0 && numH > 0) {
      const newAspectRatio = numW / numH;
      lastSyncedRatio.current = newAspectRatio;
      setDisplayPresetId('custom');
      if (!adjustments?.aspectRatio || Math.abs(adjustments.aspectRatio - newAspectRatio) > RATIO_TOLERANCE) {
        preferredPresetIdRef.current = 'custom';
        applyAspectRatio(newAspectRatio);
      } else {
        preferredPresetIdRef.current = null;
      }
    }
  };

  const handleCustomInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleApplyCustomRatio();
      (e.target as HTMLInputElement).blur();
    } else if (e.key === 'Escape') {
      setIsEditingCustom(false);
      const customRatio = aspectRatio ?? lastConstrainedRatioRef.current;
      if (customRatio) {
        const h = 100;
        const w = customRatio * h;
        setCustomW(w.toFixed(1).replace(/\.0$/, ''));
        setCustomH(h.toString());
      }
      (e.target as HTMLInputElement).blur();
    }
  };

  const getCurrentCropRatio = useCallback(() => {
    const currentCrop = adjustments.crop;
    if (currentCrop?.width && currentCrop?.height) {
      const cropRatio = currentCrop.width / currentCrop.height;
      if (currentCrop.unit === '%') {
        const imageRatio = getEffectiveOriginalRatio();
        return imageRatio ? cropRatio * imageRatio : null;
      }
      return cropRatio;
    }
    return getEffectiveOriginalRatio();
  }, [adjustments.crop, getEffectiveOriginalRatio]);

  const handlePresetChange = (presetId: string) => {
    setDisplayPresetId(presetId);
    preferredPresetIdRef.current = presetId;

    if (presetId === 'custom') {
      const customWidth = parseFloat(customW);
      const customHeight = parseFloat(customH);
      const enteredRatio = customWidth > 0 && customHeight > 0 ? customWidth / customHeight : null;
      const fallbackRatio = getCurrentCropRatio() ?? lastConstrainedRatioRef.current ?? BASE_RATIO;
      const customRatio = enteredRatio ?? fallbackRatio;
      setPreferPortrait(customRatio < 1);
      applyAspectRatio(customRatio);
      return;
    }

    const preset = PRESETS.find((candidate) => candidate.id === presetId);
    if (!preset) return;

    if (preset.value === ORIGINAL_RATIO) {
      applyAspectRatio(getEffectiveOriginalRatio());
      return;
    }

    const targetRatio = preset.value;
    let newAspectRatio = targetRatio;
    if (targetRatio && targetRatio !== 1) {
      newAspectRatio = preferPortrait && targetRatio > 1 ? 1 / targetRatio : targetRatio;
    }

    applyAspectRatio(newAspectRatio);
  };

  const handleOrientationToggle = useCallback(() => {
    if (aspectRatio && aspectRatio !== 1) {
      const newRatio = 1 / aspectRatio;
      setPreferPortrait(newRatio < 1);
      preferredPresetIdRef.current = displayPresetId;
      applyAspectRatio(newRatio);
    }
  }, [aspectRatio, applyAspectRatio, displayPresetId]);

  const handleAspectConstraintToggle = useCallback(() => {
    if (isAspectConstrained) {
      lastConstrainedRatioRef.current = aspectRatio;
      applyAspectRatio(null);
      return;
    }

    let nextPresetId = displayPresetId;
    let nextRatio: number | null = null;
    const preset = PRESETS.find((candidate) => candidate.id === displayPresetId);

    if (preset?.value === ORIGINAL_RATIO) {
      nextRatio = getEffectiveOriginalRatio();
    } else if (preset?.value) {
      nextRatio = preferPortrait && preset.value > 1 ? 1 / preset.value : preset.value;
    } else {
      nextRatio = getCurrentCropRatio() ?? lastConstrainedRatioRef.current ?? getEffectiveOriginalRatio();
      if (nextPresetId === 'free') {
        nextPresetId = 'custom';
        setDisplayPresetId(nextPresetId);
      }
    }

    if (nextRatio) {
      setPreferPortrait(nextRatio < 1);
      preferredPresetIdRef.current = nextPresetId;
      applyAspectRatio(nextRatio);
    }
  }, [
    PRESETS,
    applyAspectRatio,
    aspectRatio,
    displayPresetId,
    getCurrentCropRatio,
    getEffectiveOriginalRatio,
    isAspectConstrained,
    preferPortrait,
  ]);

  const handleReset = () => {
    const originalAspectRatio =
      selectedImage?.width && selectedImage?.height ? selectedImage.width / selectedImage.height : null;

    setPreferPortrait(false);
    setIsEditingCustom(false);
    lastSyncedRatio.current = null;
    updateLocalRotation(null);
    setEditor({ isStraightenActive: false });
    setDisplayPresetId('original');
    preferredPresetIdRef.current = 'original';
    lastConstrainedRatioRef.current = originalAspectRatio;

    setOverlay('thirds');

    setAdjustments((prev: Adjustments) => ({
      ...prev,
      aspectRatio: originalAspectRatio,
      crop: INITIAL_ADJUSTMENTS.crop,
      flipHorizontal: INITIAL_ADJUSTMENTS.flipHorizontal ?? false,
      flipVertical: INITIAL_ADJUSTMENTS.flipVertical ?? false,
      orientationSteps: INITIAL_ADJUSTMENTS.orientationSteps ?? 0,
      rotation: INITIAL_ADJUSTMENTS.rotation ?? 0,
      transformDistortion: INITIAL_ADJUSTMENTS.transformDistortion ?? 0,
      transformVertical: INITIAL_ADJUSTMENTS.transformVertical ?? 0,
      transformHorizontal: INITIAL_ADJUSTMENTS.transformHorizontal ?? 0,
      transformRotate: INITIAL_ADJUSTMENTS.transformRotate ?? 0,
      transformAspect: INITIAL_ADJUSTMENTS.transformAspect ?? 0,
      transformScale: INITIAL_ADJUSTMENTS.transformScale ?? 100,
      transformXOffset: INITIAL_ADJUSTMENTS.transformXOffset ?? 0,
      transformYOffset: INITIAL_ADJUSTMENTS.transformYOffset ?? 0,
      lensMaker: INITIAL_ADJUSTMENTS.lensMaker,
      lensModel: INITIAL_ADJUSTMENTS.lensModel,
      lensDistortionAmount: INITIAL_ADJUSTMENTS.lensDistortionAmount,
      lensVignetteAmount: INITIAL_ADJUSTMENTS.lensVignetteAmount,
      lensTcaAmount: INITIAL_ADJUSTMENTS.lensTcaAmount,
      lensDistortionEnabled: INITIAL_ADJUSTMENTS.lensDistortionEnabled,
      lensTcaEnabled: INITIAL_ADJUSTMENTS.lensTcaEnabled,
      lensVignetteEnabled: INITIAL_ADJUSTMENTS.lensVignetteEnabled,
      lensDistortionParams: INITIAL_ADJUSTMENTS.lensDistortionParams,
    }));
  };

  const isOrientationToggleDisabled =
    !isAspectConstrained || !aspectRatio || aspectRatio === 1 || displayPresetId === 'original';

  const fineRotation = useMemo(() => {
    return rotation || 0;
  }, [rotation]);

  const displayRotation = localRotation !== null ? localRotation : fineRotation;

  const handleFineRotationChange = (e: any) => {
    const newFineRotation = parseFloat(e.target.value);
    if (isRotationActive) {
      updateLocalRotation(newFineRotation);
    } else {
      setAdjustments((prev: Adjustments) => ({ ...prev, rotation: newFineRotation }));
    }
  };

  const handleStepRotate = (degrees: number) => {
    const increment = degrees > 0 ? 1 : 3;
    setEditor({ isStraightenActive: false });
    if (lastConstrainedRatioRef.current) {
      lastConstrainedRatioRef.current = 1 / lastConstrainedRatioRef.current;
      setPreferPortrait(lastConstrainedRatioRef.current < 1);
    }
    setAdjustments((prev: Adjustments) => {
      const newAspectRatio = prev.aspectRatio && prev.aspectRatio !== 0 ? 1 / prev.aspectRatio : null;
      const newOrientationSteps = ((prev.orientationSteps || 0) + increment) % 4;
      const newCrop =
        selectedImage?.width && selectedImage?.height
          ? calculateCenteredCrop(selectedImage.width, selectedImage.height, newOrientationSteps, newAspectRatio, 0)
          : null;
      return {
        ...prev,
        aspectRatio: newAspectRatio,
        orientationSteps: newOrientationSteps,
        rotation: 0,
        crop: newCrop,
      };
    });
  };

  const resetFineRotation = () => {
    updateLocalRotation(null);
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, rotation: 0 }));
  };

  const getOverlayTooltip = () => {
    const current = OVERLAYS.find((o) => o.id === activeOverlay);
    if (!current) return t('editor.crop.tooltips.compositionOverlay');
    const isRotatable = ['goldenSpiral', 'goldenTriangle'].includes(activeOverlay);
    const rotateHint = isRotatable ? t('editor.crop.tooltips.rotateHint') : '';
    return t('editor.crop.tooltips.overlayDetails', { name: current.name, rotateHint });
  };

  const getOrientationTooltip = () => {
    if (isOrientationToggleDisabled) {
      return t('editor.crop.tooltips.switchOrientation');
    }
    return orientation === Orientation.Vertical
      ? t('editor.crop.tooltips.switchToLandscape')
      : t('editor.crop.tooltips.switchToPortrait');
  };

  const handleDragStateChange = useCallback(
    (isDragging: boolean) => {
      if (isDragging) {
        setIsRotationActive(true);
        setEditor({ isRotationActive: true });
      } else {
        setIsRotationActive(false);
        setEditor({ isRotationActive: false });
        if (localRotationRef.current !== null) {
          const finalRot = localRotationRef.current;
          updateLocalRotation(null);
          setAdjustments((prev: Adjustments) => ({ ...prev, rotation: finalRot }));
        }
      }
    },
    [setEditor, updateLocalRotation, setAdjustments],
  );

  return (
    <div className="crop-panel flex h-full flex-col">
      <div className="develop-panel-header">
        <Text variant={TextVariants.heading}>{t('editor.crop.title')}</Text>
        <button
          aria-label={t('editor.crop.resetTooltip')}
          className="develop-panel-text-action"
          data-tooltip={t('editor.crop.resetTooltip')}
          onClick={handleReset}
          type="button"
        >
          {t('adjustments.basic.reset')}
        </button>
      </div>

      <div className="crop-panel-scroll grow overflow-y-auto">
        {selectedImage ? (
          <>
            <section className="crop-tool-section">
              <div className="crop-field">
                <label className="crop-field-label" htmlFor="crop-aspect-preset">
                  {t('editor.crop.aspectRatioHeading')}
                </label>
                <div className="crop-aspect-controls">
                  <select
                    aria-label={t('editor.crop.aspectRatioHeading')}
                    className="crop-select"
                    id="crop-aspect-preset"
                    onChange={(event) => handlePresetChange(event.target.value)}
                    value={displayPresetId}
                  >
                    {PRESETS.slice(0, 2).map((preset) => (
                      <option key={preset.id} title={preset.tooltip} value={preset.id}>
                        {preset.name}
                      </option>
                    ))}
                    <option title={t('editor.crop.presets.custom.tooltip')} value="custom">
                      {t('editor.crop.presets.custom.name')}
                    </option>
                    {PRESETS.slice(2).map((preset) => (
                      <option key={preset.id} title={preset.tooltip} value={preset.id}>
                        {preset.name}
                      </option>
                    ))}
                  </select>
                  <button
                    aria-label={getOrientationTooltip()}
                    className="crop-icon-button"
                    data-tooltip={getOrientationTooltip()}
                    disabled={isOrientationToggleDisabled}
                    onClick={handleOrientationToggle}
                    type="button"
                  >
                    {orientation === Orientation.Vertical ? (
                      <RectangleVertical aria-hidden="true" size={15} strokeWidth={1.8} />
                    ) : (
                      <RectangleHorizontal aria-hidden="true" size={15} strokeWidth={1.8} />
                    )}
                  </button>
                  <button
                    aria-label={
                      isAspectConstrained
                        ? t('editor.crop.tooltips.unconstrainAspect')
                        : t('editor.crop.tooltips.constrainAspect')
                    }
                    aria-pressed={isAspectConstrained}
                    className={clsx('crop-icon-button', isAspectConstrained && 'is-active')}
                    data-tooltip={
                      isAspectConstrained
                        ? t('editor.crop.tooltips.unconstrainAspect')
                        : t('editor.crop.tooltips.constrainAspect')
                    }
                    onClick={handleAspectConstraintToggle}
                    type="button"
                  >
                    {isAspectConstrained ? (
                      <Lock aria-hidden="true" size={14} strokeWidth={1.8} />
                    ) : (
                      <LockOpen aria-hidden="true" size={14} strokeWidth={1.8} />
                    )}
                  </button>
                </div>
              </div>

              {isCustomActive && (
                <div className="crop-field crop-custom-field">
                  <span className="crop-field-label">{t('editor.crop.presets.custom.name')}</span>
                  <div className="crop-custom-inputs">
                    <input
                      aria-label={t('editor.crop.custom.wTooltip')}
                      className="crop-number-input"
                      data-tooltip={t('editor.crop.custom.wTooltip')}
                      min="0"
                      name="customW"
                      onBlur={handleApplyCustomRatio}
                      onChange={handleCustomInputChange}
                      onFocus={handleCustomInputFocus}
                      onKeyDown={handleCustomInputKeyDown}
                      placeholder={t('editor.crop.custom.wPlaceholder')}
                      type="number"
                      value={customW}
                    />
                    <X aria-hidden="true" className="shrink-0 text-text-secondary" size={12} />
                    <input
                      aria-label={t('editor.crop.custom.hTooltip')}
                      className="crop-number-input"
                      data-tooltip={t('editor.crop.custom.hTooltip')}
                      min="0"
                      name="customH"
                      onBlur={handleApplyCustomRatio}
                      onChange={handleCustomInputChange}
                      onFocus={handleCustomInputFocus}
                      onKeyDown={handleCustomInputKeyDown}
                      placeholder={t('editor.crop.custom.hPlaceholder')}
                      type="number"
                      value={customH}
                    />
                  </div>
                </div>
              )}

              <div className="crop-field">
                <label className="crop-field-label" htmlFor="crop-overlay-preset">
                  {t('editor.crop.tooltips.compositionOverlay')}
                </label>
                <div className="crop-overlay-controls">
                  <select
                    aria-label={t('editor.crop.tooltips.compositionOverlay')}
                    className="crop-select"
                    id="crop-overlay-preset"
                    onChange={(event) => setOverlay(event.target.value as OverlayMode)}
                    value={activeOverlay}
                  >
                    {OVERLAYS.map((overlay) => (
                      <option key={overlay.id} title={overlay.tooltip} value={overlay.id}>
                        {overlay.name}
                      </option>
                    ))}
                  </select>
                  <button
                    aria-label={getOverlayTooltip()}
                    className="crop-icon-button"
                    data-tooltip={getOverlayTooltip()}
                    disabled={!['goldenSpiral', 'goldenTriangle'].includes(activeOverlay)}
                    onClick={() => setOverlayRotation((previous) => (previous + 1) % 4)}
                    type="button"
                  >
                    <RotateCw aria-hidden="true" size={14} strokeWidth={1.8} />
                  </button>
                </div>
              </div>

              <div className="crop-straighten-row">
                <button
                  aria-label={t('editor.crop.tooltips.straighten')}
                  aria-pressed={isStraightenActive}
                  className={clsx('crop-inline-tool', isStraightenActive && 'is-active')}
                  data-tooltip={t('editor.crop.tooltips.straighten')}
                  onClick={() => {
                    updateLocalRotation(null);
                    setEditor((state) => ({ isStraightenActive: !state.isStraightenActive }));
                  }}
                  type="button"
                >
                  <Ruler aria-hidden="true" size={14} strokeWidth={1.8} />
                  <span>{t('editor.crop.tooltips.straighten')}</span>
                </button>
                <button
                  aria-label={t('editor.crop.tooltips.resetFineRotation')}
                  className="crop-icon-button"
                  data-tooltip={t('editor.crop.tooltips.resetFineRotation')}
                  disabled={displayRotation === 0}
                  onClick={resetFineRotation}
                  type="button"
                >
                  <RotateCcw aria-hidden="true" size={14} strokeWidth={1.8} />
                </button>
              </div>

              <div className="crop-angle-slider">
                <Slider
                  defaultValue={0}
                  displayDecimals={1}
                  label={t('editor.crop.angleLabel')}
                  max={45}
                  min={-45}
                  onChange={handleFineRotationChange}
                  onDragStateChange={handleDragStateChange}
                  step={0.1}
                  suffix="°"
                  value={displayRotation}
                />
              </div>
            </section>

            <section className="crop-tool-section">
              <div className="crop-section-heading">{t('editor.crop.orientationHeading')}</div>
              <div className="crop-action-strip">
                <button
                  aria-label={t('editor.crop.tooltips.rotateLeft')}
                  className="crop-action-button"
                  data-tooltip={t('editor.crop.tooltips.rotateLeft')}
                  onClick={() => handleStepRotate(-90)}
                  type="button"
                >
                  <RotateCcw aria-hidden="true" size={17} strokeWidth={1.8} />
                </button>
                <button
                  aria-label={t('editor.crop.tooltips.rotateRight')}
                  className="crop-action-button"
                  data-tooltip={t('editor.crop.tooltips.rotateRight')}
                  onClick={() => handleStepRotate(90)}
                  type="button"
                >
                  <RotateCw aria-hidden="true" size={17} strokeWidth={1.8} />
                </button>
                <button
                  aria-label={t('editor.crop.tooltips.flipHoriz')}
                  aria-pressed={flipHorizontal}
                  className={clsx('crop-action-button', flipHorizontal && 'is-active')}
                  data-tooltip={t('editor.crop.tooltips.flipHoriz')}
                  onClick={() =>
                    setAdjustments((prev: Adjustments) => ({
                      ...prev,
                      flipHorizontal: !prev.flipHorizontal,
                    }))
                  }
                  type="button"
                >
                  <FlipHorizontal aria-hidden="true" size={17} strokeWidth={1.8} />
                </button>
                <button
                  aria-label={t('editor.crop.tooltips.flipVert')}
                  aria-pressed={flipVertical}
                  className={clsx('crop-action-button', flipVertical && 'is-active')}
                  data-tooltip={t('editor.crop.tooltips.flipVert')}
                  onClick={() => setAdjustments((prev: Adjustments) => ({ ...prev, flipVertical: !prev.flipVertical }))}
                  type="button"
                >
                  <FlipVertical aria-hidden="true" size={17} strokeWidth={1.8} />
                </button>
              </div>
            </section>

            <section className="crop-tool-section">
              <div className="crop-section-heading">{t('editor.crop.geometryHeading')}</div>
              <div className="crop-wide-actions">
                <button
                  className="crop-wide-action"
                  data-tooltip={t('editor.crop.tooltips.transform')}
                  onClick={() => setIsTransformModalOpen(true)}
                  type="button"
                >
                  <Scan aria-hidden="true" size={15} strokeWidth={1.8} />
                  <span>{t('editor.crop.labels.transform')}</span>
                </button>
                <button
                  className="crop-wide-action"
                  data-tooltip={t('editor.crop.tooltips.lens')}
                  onClick={() => setIsLensModalOpen(true)}
                  type="button"
                >
                  <Aperture aria-hidden="true" size={15} strokeWidth={1.8} />
                  <span>{t('editor.crop.labels.lens')}</span>
                </button>
              </div>
            </section>
          </>
        ) : (
          <div className="flex h-full items-center justify-center px-5">
            <Text
              className="text-center"
              color={TextColors.secondary}
              variant={TextVariants.heading}
              weight={TextWeights.normal}
            >
              {t('editor.ai.noImageSelected')}
            </Text>
          </div>
        )}
      </div>

      <TransformModal
        isOpen={isTransformModalOpen}
        onClose={() => setIsTransformModalOpen(false)}
        onApply={(newParams) => {
          setAdjustments((prev: Adjustments) => ({
            ...prev,
            transformDistortion: newParams.distortion,
            transformVertical: newParams.vertical,
            transformHorizontal: newParams.horizontal,
            transformRotate: newParams.rotate,
            transformAspect: newParams.aspect,
            transformScale: newParams.scale,
            transformXOffset: newParams.x_offset,
            transformYOffset: newParams.y_offset,
          }));
        }}
        currentAdjustments={adjustments}
      />

      <LensCorrectionModal
        isOpen={isLensModalOpen}
        onClose={() => setIsLensModalOpen(false)}
        onApply={(newParams) => {
          setAdjustments((prev: Adjustments) => ({
            ...prev,
            ...newParams,
          }));
        }}
        currentAdjustments={adjustments}
        selectedImage={selectedImage}
      />
    </div>
  );
}
