import React, { useCallback } from 'react';
import { RotateCcw, Copy, ClipboardPaste } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import CameraRawBasic from '../../adjustments/CameraRawBasic';
import CurveGraph from '../../adjustments/Curves';
import ColorPanel from '../../adjustments/Color';
import DetailsPanel from '../../adjustments/Details';
import EffectsPanel from '../../adjustments/Effects';
import GeometryPanel from '../../adjustments/Geometry';
import OpticsPanel from '../../adjustments/Optics';
import CollapsibleSection from '../../ui/CollapsibleSection';
import {
  Adjustments,
  SectionVisibility,
  INITIAL_ADJUSTMENTS,
  CAMERA_RAW_ADJUSTMENT_SECTIONS,
} from '../../../utils/adjustments';
import { useContextMenu } from '../../../context/ContextMenuContext';
import { OPTION_SEPARATOR } from '../../ui/AppProperties';
import Text from '../../ui/Text';
import { TextVariants, TextColors, TextWeights } from '../../../types/typography';
import { useShallow } from 'zustand/react/shallow';
import { useEditorStore } from '../../../store/useEditorStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { useUIStore } from '../../../store/useUIStore';
import { useEditorActions } from '../../../hooks/useEditorActions';
import { useAutoLensProfile } from '../../../hooks/useAutoLensProfile';

const CAMERA_RAW_SECTIONS = [
  { name: 'basic', titleKey: 'adjustments.basic.light' },
  { name: 'color', titleKey: 'editor.adjustments.sections.color' },
  { name: 'effects', titleKey: 'editor.adjustments.sections.effects' },
  { name: 'curves', titleKey: 'editor.adjustments.sections.curves' },
  { name: 'colorMixer', titleKey: 'editor.adjustments.sections.colorMixer' },
  { name: 'colorGrading', titleKey: 'editor.adjustments.sections.colorGrading' },
  { name: 'details', titleKey: 'editor.adjustments.sections.details' },
  { name: 'optics', titleKey: 'editor.adjustments.sections.optics' },
  { name: 'lensBlur', titleKey: 'adjustments.effects.lensBlur' },
  { name: 'geometry', titleKey: 'editor.adjustments.sections.geometry' },
  { name: 'calibration', titleKey: 'editor.adjustments.sections.calibration' },
] as const;

type CameraRawSectionName = (typeof CAMERA_RAW_SECTIONS)[number]['name'];

export default function Controls() {
  const { t } = useTranslation();
  const { showContextMenu } = useContextMenu();
  const { setAdjustments, handleAutoAdjustments, handleLutSelect, setLutPreviewOverride } = useEditorActions();

  const { appSettings, theme } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      theme: state.theme,
    })),
  );

  const { collapsibleSectionsState, setUI } = useUIStore(
    useShallow((state) => ({
      collapsibleSectionsState: state.collapsibleSectionsState,
      setUI: state.setUI,
    })),
  );

  const { adjustments, copiedSectionAdjustments, histogram, selectedImage, isWbPickerActive, setEditor } =
    useEditorStore(
      useShallow((state) => ({
        adjustments: state.adjustments,
        copiedSectionAdjustments: state.copiedSectionAdjustments,
        histogram: state.histogram,
        selectedImage: state.selectedImage,
        isWbPickerActive: state.isWbPickerActive,
        setEditor: state.setEditor,
      })),
    );

  const setCopiedSectionAdjustments = useCallback(
    (val: any) => setEditor({ copiedSectionAdjustments: val }),
    [setEditor],
  );

  const toggleWbPicker = useCallback(
    () => setEditor((state) => ({ isWbPickerActive: !state.isWbPickerActive })),
    [setEditor],
  );

  const onDragStateChange = useCallback(
    (isDragging: boolean) => setEditor({ isSliderDragging: isDragging }),
    [setEditor],
  );
  const lensProfileStatus = useAutoLensProfile({ adjustments, selectedImage, setAdjustments });

  const setCollapsibleState = useCallback(
    (updater: any) =>
      setUI((state) => ({
        collapsibleSectionsState: typeof updater === 'function' ? updater(state.collapsibleSectionsState) : updater,
      })),
    [setUI],
  );

  const handleToggleVisibility = (sectionName: string) => {
    setAdjustments((prev: Adjustments) => {
      const currentVisibility: SectionVisibility = prev.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility;
      return {
        ...prev,
        sectionVisibility: {
          ...currentVisibility,
          [sectionName]: !(currentVisibility[sectionName] ?? true),
        },
      };
    });
  };

  const handleResetAdjustments = () => {
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      ...Object.keys(CAMERA_RAW_ADJUSTMENT_SECTIONS)
        .flatMap((s) => CAMERA_RAW_ADJUSTMENT_SECTIONS[s])
        .reduce((acc: any, key: string) => {
          acc[key] = INITIAL_ADJUSTMENTS[key as keyof Adjustments];
          return acc;
        }, {}),
      sectionVisibility: { ...INITIAL_ADJUSTMENTS.sectionVisibility },
    }));
  };

  const handleToggleSection = (section: string) => {
    setCollapsibleState((prev: any) => {
      const isOpening = !prev[section];
      if (appSettings?.enableFocusMode && isOpening) {
        const newState = { ...prev };
        Object.keys(newState).forEach((key) => {
          newState[key] = false;
        });
        newState[section] = true;
        return newState;
      }
      return { ...prev, [section]: !prev[section] };
    });
  };

  const handleSectionContextMenu = (event: any, sectionName: CameraRawSectionName, titleKey: string) => {
    event.preventDefault();
    event.stopPropagation();

    const sectionKeys = CAMERA_RAW_ADJUSTMENT_SECTIONS[sectionName];
    if (!sectionKeys) {
      return;
    }

    const handleCopy = () => {
      const adjustmentsToCopy: any = {};
      for (const key of sectionKeys) {
        if (Object.prototype.hasOwnProperty.call(adjustments, key)) {
          adjustmentsToCopy[key] = JSON.parse(JSON.stringify(adjustments[key as keyof Adjustments]));
        }
      }
      setCopiedSectionAdjustments({ section: sectionName, values: adjustmentsToCopy });
    };

    const handlePaste = () => {
      if (!copiedSectionAdjustments || copiedSectionAdjustments.section !== sectionName) {
        return;
      }
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        ...copiedSectionAdjustments.values,
        sectionVisibility: {
          ...(prev.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility),
          [sectionName]: true,
        },
      }));
    };

    const handleReset = () => {
      const resetValues: any = {};
      for (const key of sectionKeys) {
        resetValues[key] = JSON.parse(JSON.stringify(INITIAL_ADJUSTMENTS[key as keyof Adjustments]));
      }
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        ...resetValues,
        sectionVisibility: {
          ...(prev.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility),
          [sectionName]: true,
        },
      }));
    };

    const isPasteAllowed = copiedSectionAdjustments && copiedSectionAdjustments.section === sectionName;
    const translatedSection = t(titleKey as never) as string;

    const pasteLabel = copiedSectionAdjustments
      ? t('editor.adjustments.actions.pasteLabel', { section: translatedSection })
      : t('editor.adjustments.actions.pasteSettings');

    const options: any = [
      {
        label: t('editor.adjustments.actions.copySectionSettings', { section: translatedSection }),
        icon: Copy,
        onClick: handleCopy,
      },
      { label: pasteLabel, icon: ClipboardPaste, onClick: handlePaste, disabled: !isPasteAllowed },
      { type: OPTION_SEPARATOR },
      {
        label: t('editor.adjustments.actions.resetSectionSettings', { section: translatedSection }),
        icon: RotateCcw,
        onClick: handleReset,
      },
    ];

    showContextMenu(event.clientX, event.clientY, options);
  };

  const renderSection = (sectionName: CameraRawSectionName) => {
    const commonProps = {
      adjustments,
      setAdjustments,
      appSettings,
      onDragStateChange,
    };

    switch (sectionName) {
      case 'basic':
        return (
          <CameraRawBasic
            {...commonProps}
            isWbPickerActive={isWbPickerActive}
            toggleWbPicker={toggleWbPicker}
            variant="light"
          />
        );
      case 'color':
        return (
          <CameraRawBasic
            {...commonProps}
            isWbPickerActive={isWbPickerActive}
            selectedImage={selectedImage}
            toggleWbPicker={toggleWbPicker}
            variant="color"
          />
        );
      case 'curves':
        return <CurveGraph {...commonProps} histogram={histogram} theme={theme} />;
      case 'details':
        return <DetailsPanel {...commonProps} variant="detail" />;
      case 'colorMixer':
        return <ColorPanel {...commonProps} variant="mixer" />;
      case 'colorGrading':
        return <ColorPanel {...commonProps} variant="grading" />;
      case 'optics':
        return <OpticsPanel {...commonProps} lensProfileStatus={lensProfileStatus} />;
      case 'geometry':
        return <GeometryPanel {...commonProps} />;
      case 'effects':
        return (
          <EffectsPanel
            {...commonProps}
            handleLutSelect={handleLutSelect}
            onLutHover={setLutPreviewOverride}
            variant="effects"
          />
        );
      case 'lensBlur':
        return (
          <EffectsPanel
            {...commonProps}
            handleLutSelect={handleLutSelect}
            onLutHover={setLutPreviewOverride}
            variant="lensBlur"
          />
        );
      case 'calibration':
        return <ColorPanel {...commonProps} variant="calibration" />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="develop-panel-header">
        <div className="min-w-0">
          <Text variant={TextVariants.heading}>{t('editor.adjustments.title')}</Text>
        </div>
        <div className="develop-panel-header-actions">
          <button
            aria-label={t('editor.adjustments.tooltips.autoAdjust')}
            className="develop-panel-text-action"
            disabled={!selectedImage}
            onClick={handleAutoAdjustments}
            data-tooltip={t('editor.adjustments.tooltips.autoAdjust')}
            type="button"
          >
            {t('settings.processing.backends.auto')}
          </button>
          <button
            aria-label={t('editor.adjustments.tooltips.resetAdjustments')}
            className="develop-panel-text-action"
            disabled={!selectedImage}
            onClick={handleResetAdjustments}
            data-tooltip={t('editor.adjustments.tooltips.resetAdjustments')}
            type="button"
          >
            {t('adjustments.basic.reset')}
          </button>
        </div>
      </div>

      {!appSettings?.tonemapperOverrideEnabled && (
        <label className="develop-profile-strip">
          <span>{t('adjustments.optics.profile')}</span>
          <select
            aria-label={t('adjustments.optics.profile')}
            className="camera-raw-select"
            onChange={(event) =>
              setAdjustments((previous: Adjustments) => ({
                ...previous,
                toneMapper: event.target.value as 'basic' | 'agx',
              }))
            }
            value={adjustments.toneMapper || 'basic'}
          >
            <option value="basic">{t('adjustments.basic.mappers.basic')}</option>
            <option value="agx">{t('adjustments.basic.mappers.agx')}</option>
          </select>
        </label>
      )}

      <div className="develop-adjustment-stack custom-scrollbar">
        {selectedImage ? (
          CAMERA_RAW_SECTIONS.map(({ name: sectionName, titleKey }) => {
            const title = t(titleKey as never) as string;
            const sectionVisibility = adjustments.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility;

            return (
              <div className="shrink-0 group" key={sectionName}>
                <CollapsibleSection
                  isContentVisible={sectionVisibility[sectionName as keyof SectionVisibility] ?? true}
                  isOpen={collapsibleSectionsState[sectionName as keyof typeof collapsibleSectionsState] ?? false}
                  onContextMenu={(e: any) => handleSectionContextMenu(e, sectionName, titleKey)}
                  onToggle={() => handleToggleSection(sectionName)}
                  onToggleVisibility={() => handleToggleVisibility(sectionName)}
                  title={title}
                >
                  {renderSection(sectionName)}
                </CollapsibleSection>
              </div>
            );
          })
        ) : (
          <div className="flex items-center justify-center h-full">
            <Text
              variant={TextVariants.heading}
              color={TextColors.secondary}
              weight={TextWeights.normal}
              className="text-center"
            >
              {t('editor.ai.noImageSelected')}
            </Text>
          </div>
        )}
      </div>
    </div>
  );
}
