import React, { useCallback } from 'react';
import { RotateCcw, Copy, ClipboardPaste, Aperture } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import BasicAdjustments from '../../adjustments/Basic';
import CurveGraph from '../../adjustments/Curves';
import ColorPanel from '../../adjustments/Color';
import DetailsPanel from '../../adjustments/Details';
import EffectsPanel from '../../adjustments/Effects';
import CollapsibleSection from '../../ui/CollapsibleSection';
import { Adjustments, SectionVisibility, INITIAL_ADJUSTMENTS, ADJUSTMENT_SECTIONS } from '../../../utils/adjustments';
import { useContextMenu } from '../../../context/ContextMenuContext';
import { OPTION_SEPARATOR } from '../../ui/AppProperties';
import Text from '../../ui/Text';
import { TextVariants, TextColors, TextWeights } from '../../../types/typography';
import { useShallow } from 'zustand/react/shallow';
import { useEditorStore } from '../../../store/useEditorStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { useUIStore } from '../../../store/useUIStore';
import { useEditorActions } from '../../../hooks/useEditorActions';
import { BASIC_MODE } from '../../../basic/runtime';

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
          [sectionName]: !currentVisibility[sectionName],
        },
      };
    });
  };

  const handleResetAdjustments = () => {
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      ...Object.keys(ADJUSTMENT_SECTIONS)
        .flatMap((s) => ADJUSTMENT_SECTIONS[s])
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

  const handleSectionContextMenu = (event: any, sectionName: string) => {
    event.preventDefault();
    event.stopPropagation();

    const sectionKeys = ADJUSTMENT_SECTIONS[sectionName];
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
    const translatedSection = t(`editor.adjustments.sections.${sectionName}` as never) as string;

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

  return (
    <div className="flex flex-col h-full">
      <div className="develop-panel-header">
        <div className="min-w-0">
          <Text variant={TextVariants.heading}>{t('editor.adjustments.title')}</Text>
          {BASIC_MODE && (
            <div className="mt-0.5 text-[10px] text-text-secondary">{t('editor.adjustments.basicMode')}</div>
          )}
        </div>
        <div className="flex items-center gap-0.5">
          <button
            aria-label={t('editor.adjustments.tooltips.autoAdjust')}
            className="develop-panel-action"
            disabled={!selectedImage}
            onClick={handleAutoAdjustments}
            data-tooltip={t('editor.adjustments.tooltips.autoAdjust')}
          >
            <Aperture size={16} strokeWidth={1.8} />
          </button>
          <button
            aria-label={t('editor.adjustments.tooltips.resetAdjustments')}
            className="develop-panel-action"
            disabled={!selectedImage}
            onClick={handleResetAdjustments}
            data-tooltip={t('editor.adjustments.tooltips.resetAdjustments')}
          >
            <RotateCcw size={16} strokeWidth={1.8} />
          </button>
        </div>
      </div>

      <div className="develop-adjustment-stack custom-scrollbar">
        {selectedImage ? (
          Object.keys(ADJUSTMENT_SECTIONS).map((sectionName: string) => {
            const SectionComponent: any = {
              basic: BasicAdjustments,
              curves: CurveGraph,
              color: ColorPanel,
              details: DetailsPanel,
              effects: EffectsPanel,
            }[sectionName];

            const title = t(`editor.adjustments.sections.${sectionName}` as never) as string;
            const sectionVisibility = adjustments.sectionVisibility || INITIAL_ADJUSTMENTS.sectionVisibility;

            return (
              <div className="shrink-0 group" key={sectionName}>
                <CollapsibleSection
                  isContentVisible={sectionVisibility[sectionName as keyof SectionVisibility]}
                  isOpen={collapsibleSectionsState[sectionName as keyof typeof collapsibleSectionsState]}
                  onContextMenu={(e: any) => handleSectionContextMenu(e, sectionName)}
                  onToggle={() => handleToggleSection(sectionName)}
                  onToggleVisibility={() => handleToggleVisibility(sectionName)}
                  title={title}
                >
                  <SectionComponent
                    adjustments={adjustments}
                    setAdjustments={setAdjustments}
                    histogram={histogram}
                    theme={theme}
                    handleLutSelect={handleLutSelect}
                    onLutHover={setLutPreviewOverride}
                    appSettings={appSettings}
                    isWbPickerActive={isWbPickerActive}
                    toggleWbPicker={toggleWbPicker}
                    onDragStateChange={onDragStateChange}
                  />
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
