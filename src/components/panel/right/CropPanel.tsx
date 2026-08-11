import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronRight,
  FlipHorizontal,
  FlipVertical,
  Loader2,
  Lock,
  LockOpen,
  RectangleHorizontal,
  RectangleVertical,
  RotateCcw,
  RotateCw,
  Ruler,
  Sparkles,
  X,
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import { Adjustments, INITIAL_ADJUSTMENTS } from '../../../utils/adjustments';
import clsx from 'clsx';
import { Invokes, Orientation } from '../../ui/AppProperties';
import Text from '../../ui/Text';
import Slider from '../../ui/Slider';
import { TextColors, TextVariants, TextWeights } from '../../../types/typography';
import { useEditorStore } from '../../../store/useEditorStore';
import { useEditorActions } from '../../../hooks/useEditorActions';
import {
  calculateAreaPreservingCrop,
  calculateCenteredCrop,
  isCropWithinBounds,
  rotateCropQuarterTurn,
} from '../../../utils/cropUtils';
import { Crop } from 'react-image-crop';
import GeometryPanel from '../../adjustments/Geometry';
import {
  getCropGuideOrientationCount,
  getNextCropGuide,
  ROTATABLE_CROP_GUIDES,
  type CropGuideMode,
} from '../../../types/crop';

const BASE_RATIO = 1.618;
const ORIGINAL_RATIO = 0;
const RATIO_TOLERANCE = 0.01;

function orientRatio(ratio: number, portrait: boolean): number {
  if (Math.abs(ratio - 1) < RATIO_TOLERANCE) return 1;
  if (portrait) return ratio > 1 ? 1 / ratio : ratio;
  return ratio < 1 ? 1 / ratio : ratio;
}

interface CropPreset {
  id: string;
  name: string;
  value: number | null;
  tooltip: string;
}

interface StraightenAnalysisResult {
  angle: number;
  confidence: number;
  detected: boolean;
  lineCount: number;
}

export default function CropPanel() {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((s) => s.selectedImage);
  const adjustments = useEditorStore((s) => s.adjustments);
  const isStraightenActive = useEditorStore((s) => s.isStraightenActive);
  const cropGuideMode = useEditorStore((s) => s.cropGuideMode);
  const setEditor = useEditorStore((s) => s.setEditor);
  const { setAdjustments } = useEditorActions();
  const [customW, setCustomW] = useState('');
  const [customH, setCustomH] = useState('');
  const [isRotationActive, setIsRotationActive] = useState(false);
  const [preferPortrait, setPreferPortrait] = useState(false);
  const [isEditingCustom, setIsEditingCustom] = useState(false);
  const [displayPresetId, setDisplayPresetId] = useState('free');
  const [isGeometryExpanded, setIsGeometryExpanded] = useState(false);
  const [isAutoStraightening, setIsAutoStraightening] = useState(false);

  const [localRotation, setLocalRotation] = useState<number | null>(null);
  const localRotationRef = useRef<number | null>(null);
  const lastConstrainedRatioRef = useRef<number | null>(null);
  const preferredPresetIdRef = useRef<string | null>(null);
  const selectedImagePathRef = useRef<string | null>(null);
  const autoStraightenRequestRef = useRef(0);

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

  const updateLocalRotation = useCallback(
    (val: number | null) => {
      setLocalRotation(val);
      localRotationRef.current = val;
      setEditor({ liveRotation: val });
    },
    [setEditor],
  );

  const setCropGuide = useCallback((mode: CropGuideMode) => setEditor({ cropGuideMode: mode }), [setEditor]);

  const rotateCropGuide = useCallback(
    (updater: React.SetStateAction<number>) => {
      setEditor((state) => ({
        cropGuideRotation: typeof updater === 'function' ? updater(state.cropGuideRotation) : updater,
      }));
    },
    [setEditor],
  );

  const lastSyncedRatio = useRef<number | null>(null);

  const {
    aspectRatio,
    constrainCrop = true,
    rotation = 0,
    flipHorizontal = false,
    flipVertical = false,
    orientationSteps = 0,
  } = adjustments;

  useEffect(() => {
    if (isStraightenActive) {
      updateLocalRotation(null);
    }
  }, [isStraightenActive, updateLocalRotation]);

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
    if (
      originalRatio &&
      (Math.abs(aspectRatio - originalRatio) < RATIO_TOLERANCE ||
        Math.abs(aspectRatio - 1 / originalRatio) < RATIO_TOLERANCE)
    ) {
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
        return (
          originalRatio !== null &&
          (Math.abs(aspectRatio - originalRatio) < RATIO_TOLERANCE ||
            Math.abs(aspectRatio - 1 / originalRatio) < RATIO_TOLERANCE)
        );
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
            constrainCrop,
          ) ??
          calculateCenteredCrop(
            selectedImage.width,
            selectedImage.height,
            orientationSteps,
            newAspectRatio,
            rotation,
            constrainCrop,
          );
      }
      setAdjustments((prev: Adjustments) => ({ ...prev, aspectRatio: newAspectRatio, crop: newCrop }));
    },
    [selectedImage, orientationSteps, rotation, adjustments.crop, constrainCrop, setAdjustments],
  );

  useEffect(() => {
    if (displayPresetId === 'original') {
      const originalRatio = getEffectiveOriginalRatio();
      const nextRatio = originalRatio === null ? null : orientRatio(originalRatio, preferPortrait);
      if (nextRatio !== null && aspectRatio && Math.abs(aspectRatio - nextRatio) > RATIO_TOLERANCE) {
        preferredPresetIdRef.current = 'original';
        applyAspectRatio(nextRatio);
      }
    }
  }, [orientationSteps, displayPresetId, aspectRatio, getEffectiveOriginalRatio, applyAspectRatio, preferPortrait]);

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
      const originalRatio = getEffectiveOriginalRatio();
      if (originalRatio) {
        setPreferPortrait(originalRatio < 1);
        applyAspectRatio(originalRatio);
      }
      return;
    }

    const targetRatio = preset.value;
    const newAspectRatio = targetRatio ? orientRatio(targetRatio, preferPortrait) : null;

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
      const originalRatio = getEffectiveOriginalRatio();
      nextRatio = originalRatio ? orientRatio(originalRatio, preferPortrait) : null;
    } else if (preset?.value) {
      nextRatio = orientRatio(preset.value, preferPortrait);
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

  const handleResetCrop = useCallback(() => {
    const originalAspectRatio =
      selectedImage?.width && selectedImage?.height ? selectedImage.width / selectedImage.height : null;

    setPreferPortrait(Boolean(originalAspectRatio && originalAspectRatio < 1));
    setIsEditingCustom(false);
    lastSyncedRatio.current = null;
    updateLocalRotation(null);
    setEditor({ isStraightenActive: false });
    setDisplayPresetId('original');
    preferredPresetIdRef.current = 'original';
    lastConstrainedRatioRef.current = originalAspectRatio;

    setAdjustments((prev: Adjustments) => ({
      ...prev,
      aspectRatio: originalAspectRatio,
      constrainCrop: INITIAL_ADJUSTMENTS.constrainCrop,
      crop: INITIAL_ADJUSTMENTS.crop,
      flipHorizontal: INITIAL_ADJUSTMENTS.flipHorizontal ?? false,
      flipVertical: INITIAL_ADJUSTMENTS.flipVertical ?? false,
      orientationSteps: INITIAL_ADJUSTMENTS.orientationSteps ?? 0,
      rotation: INITIAL_ADJUSTMENTS.rotation ?? 0,
    }));
  }, [selectedImage, setAdjustments, setEditor, updateLocalRotation]);

  const isOrientationToggleDisabled = !isAspectConstrained || !aspectRatio || aspectRatio === 1;

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
    const direction = degrees > 0 ? 1 : -1;
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
          ? prev.crop
            ? rotateCropQuarterTurn(
                prev.crop,
                selectedImage.width,
                selectedImage.height,
                prev.orientationSteps || 0,
                direction,
              )
            : calculateCenteredCrop(
                selectedImage.width,
                selectedImage.height,
                newOrientationSteps,
                newAspectRatio,
                0,
                prev.constrainCrop ?? true,
              )
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

  const toggleStraighten = useCallback(() => {
    updateLocalRotation(null);
    setEditor((state) => ({ isStraightenActive: !state.isStraightenActive }));
  }, [setEditor, updateLocalRotation]);

  const handleAutoStraighten = useCallback(async () => {
    if (!selectedImage || isAutoStraightening) return;

    const requestId = ++autoStraightenRequestRef.current;
    const imagePath = selectedImage.path;
    setIsAutoStraightening(true);
    setEditor({ isStraightenActive: false });
    updateLocalRotation(null);

    try {
      const result = await invoke<StraightenAnalysisResult>(Invokes.AnalyzeCropStraighten, {
        jsAdjustments: adjustments,
      });

      if (requestId !== autoStraightenRequestRef.current || selectedImagePathRef.current !== imagePath) return;

      if (!result.detected) {
        toast.info(t('editor.crop.autoStraightenNoLines'));
        return;
      }

      setAdjustments((prev: Adjustments) => ({ ...prev, rotation: result.angle }));
    } catch (error) {
      if (requestId === autoStraightenRequestRef.current) {
        toast.error(t('editor.crop.autoStraightenFailed', { error: String(error) }));
      }
    } finally {
      if (requestId === autoStraightenRequestRef.current) {
        setIsAutoStraightening(false);
      }
    }
  }, [adjustments, isAutoStraightening, selectedImage, setAdjustments, setEditor, t, updateLocalRotation]);

  useEffect(
    () => () => {
      autoStraightenRequestRef.current += 1;
    },
    [],
  );

  const cropDimensionsLabel = useMemo(() => {
    if (!selectedImage?.width || !selectedImage?.height) return '';
    const isSwapped = orientationSteps === 1 || orientationSteps === 3;
    const fallbackWidth = isSwapped ? selectedImage.height : selectedImage.width;
    const fallbackHeight = isSwapped ? selectedImage.width : selectedImage.height;
    const width = Math.max(1, Math.round(adjustments.crop?.width ?? fallbackWidth));
    const height = Math.max(1, Math.round(adjustments.crop?.height ?? fallbackHeight));
    return t('editor.crop.outputDimensions', { width, height });
  }, [adjustments.crop, orientationSteps, selectedImage, t]);

  const hasGeometryAdjustments = useMemo(
    () =>
      (adjustments.transformDistortion ?? 0) !== (INITIAL_ADJUSTMENTS.transformDistortion ?? 0) ||
      (adjustments.transformVertical ?? 0) !== (INITIAL_ADJUSTMENTS.transformVertical ?? 0) ||
      (adjustments.transformHorizontal ?? 0) !== (INITIAL_ADJUSTMENTS.transformHorizontal ?? 0) ||
      (adjustments.transformRotate ?? 0) !== (INITIAL_ADJUSTMENTS.transformRotate ?? 0) ||
      (adjustments.transformAspect ?? 0) !== (INITIAL_ADJUSTMENTS.transformAspect ?? 0) ||
      (adjustments.transformScale ?? 100) !== (INITIAL_ADJUSTMENTS.transformScale ?? 100) ||
      (adjustments.transformXOffset ?? 0) !== (INITIAL_ADJUSTMENTS.transformXOffset ?? 0) ||
      (adjustments.transformYOffset ?? 0) !== (INITIAL_ADJUSTMENTS.transformYOffset ?? 0),
    [adjustments],
  );

  const resetGeometry = useCallback(() => {
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      transformDistortion: INITIAL_ADJUSTMENTS.transformDistortion ?? 0,
      transformVertical: INITIAL_ADJUSTMENTS.transformVertical ?? 0,
      transformHorizontal: INITIAL_ADJUSTMENTS.transformHorizontal ?? 0,
      transformRotate: INITIAL_ADJUSTMENTS.transformRotate ?? 0,
      transformAspect: INITIAL_ADJUSTMENTS.transformAspect ?? 0,
      transformScale: INITIAL_ADJUSTMENTS.transformScale ?? 100,
      transformXOffset: INITIAL_ADJUSTMENTS.transformXOffset ?? 0,
      transformYOffset: INITIAL_ADJUSTMENTS.transformYOffset ?? 0,
    }));
  }, [setAdjustments]);

  const handleGeometryDragStateChange = useCallback(
    (isDragging: boolean) => setEditor({ isSliderDragging: isDragging }),
    [setEditor],
  );

  const nudgeCrop = useCallback(
    (deltaX: number, deltaY: number) => {
      if (!selectedImage?.width || !selectedImage?.height) return;

      setAdjustments((prev: Adjustments) => {
        if (!prev.crop?.width || !prev.crop?.height) return prev;

        const isSwapped = prev.orientationSteps === 1 || prev.orientationSteps === 3;
        const imageWidth = isSwapped ? selectedImage.height : selectedImage.width;
        const imageHeight = isSwapped ? selectedImage.width : selectedImage.height;
        const sourceCrop =
          prev.crop.unit === '%'
            ? {
                unit: 'px' as const,
                x: (prev.crop.x / 100) * imageWidth,
                y: (prev.crop.y / 100) * imageHeight,
                width: (prev.crop.width / 100) * imageWidth,
                height: (prev.crop.height / 100) * imageHeight,
              }
            : { ...prev.crop, unit: 'px' as const };
        const desired = {
          ...sourceCrop,
          unit: 'px' as const,
          x: Math.min(imageWidth - sourceCrop.width, Math.max(0, sourceCrop.x + deltaX)),
          y: Math.min(imageHeight - sourceCrop.height, Math.max(0, sourceCrop.y + deltaY)),
        };

        if (isCropWithinBounds(desired, imageWidth, imageHeight, prev.rotation || 0, prev.constrainCrop ?? true)) {
          return { ...prev, crop: desired };
        }

        let low = 0;
        let high = 1;
        let best = sourceCrop;
        for (let index = 0; index < 12; index += 1) {
          const amount = (low + high) / 2;
          const candidate = {
            ...sourceCrop,
            unit: 'px' as const,
            x: sourceCrop.x + (desired.x - sourceCrop.x) * amount,
            y: sourceCrop.y + (desired.y - sourceCrop.y) * amount,
          };
          if (
            isCropWithinBounds(candidate, imageWidth, imageHeight, prev.rotation || 0, prev.constrainCrop ?? true)
          ) {
            best = candidate;
            low = amount;
          } else {
            high = amount;
          }
        }

        const rounded = { ...best, x: Math.round(best.x), y: Math.round(best.y) };
        return rounded.x === sourceCrop.x && rounded.y === sourceCrop.y ? prev : { ...prev, crop: rounded };
      });
    },
    [selectedImage, setAdjustments],
  );

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

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isEditingText =
        target?.matches('input, textarea, select, [contenteditable="true"]') ||
        Boolean(target?.closest('[contenteditable="true"]'));
      const isArrowKey = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key);
      if (isEditingText || (event.repeat && !isArrowKey)) return;

      const hasCommandModifier = event.metaKey || event.ctrlKey;

      if (hasCommandModifier && event.altKey && event.code === 'KeyR') {
        event.preventDefault();
        handleResetCrop();
        return;
      }
      if (hasCommandModifier) return;

      if (event.altKey && event.code === 'KeyV') {
        event.preventDefault();
        setCropGuide(getNextCropGuide(cropGuideMode));
        return;
      }
      if (event.shiftKey && !event.altKey && event.code === 'KeyV') {
        if (ROTATABLE_CROP_GUIDES.has(cropGuideMode)) {
          event.preventDefault();
          const orientationCount = getCropGuideOrientationCount(cropGuideMode);
          rotateCropGuide((current) => (current + 1) % orientationCount);
        }
        return;
      }
      if (event.altKey && event.code === 'KeyA') {
        event.preventDefault();
        handleAspectConstraintToggle();
        return;
      }
      if (!event.altKey && !event.shiftKey && event.code === 'KeyX') {
        event.preventDefault();
        handleAspectConstraintToggle();
        return;
      }
      if (!event.altKey && isArrowKey) {
        event.preventDefault();
        const distance = event.shiftKey ? 10 : 1;
        const deltaX = event.key === 'ArrowLeft' ? -distance : event.key === 'ArrowRight' ? distance : 0;
        const deltaY = event.key === 'ArrowUp' ? -distance : event.key === 'ArrowDown' ? distance : 0;
        nudgeCrop(deltaX, deltaY);
        return;
      }
      if (event.key === 'Escape' && isStraightenActive) {
        event.preventDefault();
        setEditor({ isStraightenActive: false });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    cropGuideMode,
    handleAspectConstraintToggle,
    handleResetCrop,
    isStraightenActive,
    nudgeCrop,
    rotateCropGuide,
    setCropGuide,
    setEditor,
  ]);

  return (
    <div className="crop-panel flex h-full flex-col">
      <div className="develop-panel-header">
        <Text variant={TextVariants.heading}>{t('editor.crop.title')}</Text>
        <button
          aria-label={t('editor.crop.resetTooltip')}
          className="develop-panel-text-action"
          data-tooltip={t('editor.crop.resetTooltip')}
          onClick={handleResetCrop}
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

              <div className="crop-straighten-row">
                <div className="crop-inline-tool-group">
                  <button
                    aria-label={t('editor.crop.autoStraighten')}
                    className="crop-inline-tool"
                    data-tooltip={t('editor.crop.autoStraightenTooltip')}
                    disabled={isAutoStraightening}
                    onClick={handleAutoStraighten}
                    type="button"
                  >
                    {isAutoStraightening ? (
                      <Loader2 aria-hidden="true" className="animate-spin" size={14} strokeWidth={1.8} />
                    ) : (
                      <Sparkles aria-hidden="true" size={14} strokeWidth={1.8} />
                    )}
                    <span>{t('editor.crop.autoStraighten')}</span>
                  </button>
                  <button
                    aria-label={t('editor.crop.tooltips.straighten')}
                    aria-pressed={isStraightenActive}
                    className={clsx('crop-inline-tool', isStraightenActive && 'is-active')}
                    data-tooltip={t('editor.crop.tooltips.straighten')}
                    onClick={toggleStraighten}
                    type="button"
                  >
                    <Ruler aria-hidden="true" size={14} strokeWidth={1.8} />
                    <span>{t('editor.crop.tooltips.straighten')}</span>
                  </button>
                </div>
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

              <div className="crop-constraint-row">
                <label className="crop-checkbox-label">
                  <input
                    checked={constrainCrop}
                    onChange={(event) =>
                      setAdjustments((prev: Adjustments) => ({ ...prev, constrainCrop: event.target.checked }))
                    }
                    type="checkbox"
                  />
                  <span>{t('editor.crop.constrainToImage')}</span>
                </label>
                <output className="crop-dimensions" title={t('editor.crop.outputDimensionsTooltip')}>
                  {cropDimensionsLabel}
                </output>
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

            <section className="crop-tool-section crop-geometry-section">
              <div className="crop-collapsible-heading">
                <button
                  aria-expanded={isGeometryExpanded}
                  className="crop-collapsible-trigger"
                  onClick={() => setIsGeometryExpanded((expanded) => !expanded)}
                  type="button"
                >
                  <ChevronRight
                    aria-hidden="true"
                    className={clsx('crop-collapsible-chevron', isGeometryExpanded && 'is-expanded')}
                    size={14}
                    strokeWidth={2}
                  />
                  <span>{t('editor.crop.geometryHeading')}</span>
                </button>
                <button
                  aria-label={t('modals.transform.resetTooltip')}
                  className="crop-icon-button"
                  data-tooltip={t('modals.transform.resetTooltip')}
                  disabled={!hasGeometryAdjustments}
                  onClick={resetGeometry}
                  type="button"
                >
                  <RotateCcw aria-hidden="true" size={14} strokeWidth={1.8} />
                </button>
              </div>
              <div
                aria-hidden={!isGeometryExpanded}
                className={clsx('crop-collapsible-content', isGeometryExpanded && 'is-expanded')}
                inert={!isGeometryExpanded}
              >
                <div>
                  <GeometryPanel
                    adjustments={adjustments}
                    compact
                    onDragStateChange={handleGeometryDragStateChange}
                    setAdjustments={setAdjustments}
                  />
                </div>
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
    </div>
  );
}
