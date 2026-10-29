import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { ClipboardPaste, Copy, Folder as FolderIcon, RotateCcw } from 'lucide-react';

import CollapsibleSection from '../../../ui/CollapsibleSection';
import Switch from '../../../ui/Switch';
import Slider from '../../../ui/Slider';
import TaskProgress from '../../../ui/TaskProgress';
import BasicAdjustments from '../../../adjustments/Basic';
import CurveGraph from '../../../adjustments/Curves';
import ColorPanel from '../../../adjustments/Color';
import DetailsPanel from '../../../adjustments/Details';
import EffectsPanel from '../../../adjustments/Effects';
import { DepthRangePicker } from '../../../ui/DepthRangePicker';

import { Mask, getSubMaskName } from '../Masks';
import {
  Adjustments,
  INITIAL_MASK_ADJUSTMENTS,
  INITIAL_MASK_CONTAINER,
  ADJUSTMENT_SECTIONS,
} from '../../../../utils/adjustments';
import { useContextMenu } from '../../../../context/ContextMenuContext';
import { OPTION_SEPARATOR } from '../../../ui/AppProperties';
import Text from '../../../ui/Text';
import { TextVariants } from '../../../../types/typography';
import { BrushTools, FlowBrushTool } from './BrushTools';
import { SUB_MASK_CONFIG } from './subMaskConfig';

export function SettingsPanel({
  container,
  activeSubMask,
  aiModelDownloadStatus,
  brushSettings,
  setBrushSettings,
  updateContainer,
  updateSubMask,
  histogram,
  appSettings,
  isGeneratingAiMask,
  setIsMaskControlHovered,
  collapsibleState,
  setCollapsibleState,
  copiedSectionAdjustments,
  setCopiedSectionAdjustments,
  onDragStateChange,
  isSettingsSectionOpen,
  setSettingsSectionOpen,
  presets,
}: any) {
  const { t } = useTranslation();
  const { showContextMenu } = useContextMenu();
  const isActive = !!container;
  const presetButtonRef = useRef<HTMLButtonElement>(null);

  const placeholderContainer = {
    ...INITIAL_MASK_CONTAINER,
    adjustments: INITIAL_MASK_ADJUSTMENTS,
  };
  const displayContainer = container || placeholderContainer;

  const handleApplyPresetToMask = (presetAdjustments: Partial<Adjustments>) => {
    if (!container) return;
    const currentAdjustments = container.adjustments;
    const newMaskAdjustments = {
      ...currentAdjustments,
      ...presetAdjustments,
      sectionVisibility: {
        ...(currentAdjustments.sectionVisibility || INITIAL_MASK_ADJUSTMENTS.sectionVisibility),
        ...(presetAdjustments.sectionVisibility || {}),
      },
    };
    updateContainer(container.id, { adjustments: newMaskAdjustments });
  };

  const generatePresetSubmenu = (presetList: any[]): any[] => {
    return presetList
      .map((item: any) => {
        if (item.folder) {
          return {
            label: item.folder.name,
            icon: FolderIcon,
            submenu: generatePresetSubmenu(item.folder.children),
          };
        }
        if (item.preset || item.adjustments) {
          return {
            label: item.name || item.preset.name,
            onClick: () => handleApplyPresetToMask(item.adjustments || item.preset.adjustments),
          };
        }
        return null;
      })
      .filter(Boolean);
  };

  const handlePresetSelectClick = () => {
    if (presetButtonRef.current) {
      const rect = presetButtonRef.current.getBoundingClientRect();
      const presetSubmenu = generatePresetSubmenu(presets);
      const options =
        presetSubmenu.length > 0
          ? presetSubmenu
          : [{ label: t('editor.masks.settings.noPresetsFound'), disabled: true }];
      showContextMenu(rect.left, rect.bottom + 5, options);
    }
  };

  const handleMaskPropertyChange = (key: string, value: any) => {
    if (!isActive) return;
    updateContainer(container.id, { [key]: value });
  };

  const handleSubMaskParametersChange = (changes: Record<string, number>) => {
    if (!isActive || !activeSubMask) return;
    const newParams = { ...activeSubMask.parameters, ...changes };
    updateSubMask(activeSubMask.id, { parameters: newParams });
  };

  const handleDepthRangeChange = (values: { minDepth: number; maxDepth: number; minFade: number; maxFade: number }) => {
    if (!isActive || !activeSubMask) return;

    const newParams = {
      ...activeSubMask.parameters,
      minDepth: 100 - values.maxDepth,
      maxDepth: 100 - values.minDepth,
      minFade: values.maxFade,
      maxFade: values.minFade,
    };
    updateSubMask(activeSubMask.id, { parameters: newParams });
  };

  const subMaskConfig = activeSubMask ? SUB_MASK_CONFIG[activeSubMask.type as Mask] || {} : {};
  const isAiMask = activeSubMask && ['ai-subject', 'ai-foreground', 'ai-sky', 'ai-depth'].includes(activeSubMask.type);
  const isComponentMode = !!activeSubMask;

  const setMaskContainerAdjustments = (updater: any) => {
    if (!isActive) return;
    const currentAdjustments = container.adjustments;
    const newAdjustments = typeof updater === 'function' ? updater(currentAdjustments) : updater;
    updateContainer(container.id, { adjustments: newAdjustments });
  };

  const handleToggleSection = (section: string) => {
    setCollapsibleState((prev: any) => {
      const isOpening = !prev[section];
      if (appSettings?.enableFocusMode && isOpening) {
        setSettingsSectionOpen(false);
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

  const handleToggleVisibility = (sectionName: string) => {
    if (!isActive) return;
    const cur = container.adjustments;
    const vis = cur.sectionVisibility || INITIAL_MASK_ADJUSTMENTS.sectionVisibility;
    updateContainer(container.id, {
      adjustments: { ...cur, sectionVisibility: { ...vis, [sectionName]: !vis[sectionName] } },
    });
  };

  const handleSectionContextMenu = (event: any, sectionName: string) => {
    if (!isActive) return;
    event.preventDefault();
    event.stopPropagation();

    const sectionKeys = ADJUSTMENT_SECTIONS[sectionName];
    if (!sectionKeys) return;

    const handleCopy = () => {
      const adjustmentsToCopy: Record<string, any> = {};
      for (const key of sectionKeys) {
        if (container.adjustments && container.adjustments[key] !== undefined) {
          adjustmentsToCopy[key] = JSON.parse(JSON.stringify(container.adjustments[key]));
        }
      }
      setCopiedSectionAdjustments({ section: sectionName, values: adjustmentsToCopy });
    };

    const handlePaste = () => {
      if (!copiedSectionAdjustments || copiedSectionAdjustments.section !== sectionName) return;

      setMaskContainerAdjustments((prev: any) => ({
        ...prev,
        ...copiedSectionAdjustments.values,
        sectionVisibility: {
          ...(prev.sectionVisibility || INITIAL_MASK_ADJUSTMENTS.sectionVisibility),
          [sectionName]: true,
        },
      }));
    };

    const handleReset = () => {
      const resetValues: any = {};
      for (const key of sectionKeys) {
        if (INITIAL_MASK_ADJUSTMENTS[key] !== undefined) {
          resetValues[key] = JSON.parse(JSON.stringify(INITIAL_MASK_ADJUSTMENTS[key]));
        }
      }
      setMaskContainerAdjustments((prev: any) => ({
        ...prev,
        ...resetValues,
        sectionVisibility: {
          ...(prev.sectionVisibility || INITIAL_MASK_ADJUSTMENTS.sectionVisibility),
          [sectionName]: true,
        },
      }));
    };

    const isPasteAllowed = copiedSectionAdjustments && copiedSectionAdjustments.section === sectionName;
    const sectionTitle = sectionName.charAt(0).toUpperCase() + sectionName.slice(1);

    const pasteLabel = copiedSectionAdjustments
      ? t('editor.masks.settings.pasteSectionSettings', { section: sectionTitle })
      : t('editor.masks.settings.pasteSettings');

    showContextMenu(event.clientX, event.clientY, [
      {
        icon: Copy,
        label: t('editor.masks.settings.copySectionSettings', { section: sectionTitle }),
        onClick: handleCopy,
      },
      { label: pasteLabel, icon: ClipboardPaste, onClick: handlePaste, disabled: !isPasteAllowed },
      { type: OPTION_SEPARATOR },
      {
        icon: RotateCcw,
        label: t('editor.masks.settings.resetSectionSettings', { section: sectionTitle }),
        onClick: handleReset,
      },
    ]);
  };

  const sectionVisibility =
    displayContainer.adjustments.sectionVisibility || INITIAL_MASK_ADJUSTMENTS.sectionVisibility;

  return (
    <div
      className={`space-y-2 transition-opacity duration-300 ${!isActive ? 'opacity-50 pointer-events-none' : ''}`}
      onClick={(e) => e.stopPropagation()}
    >
      <CollapsibleSection
        title={
          isComponentMode
            ? t('editor.masks.settings.componentPropertiesTitle', { name: getSubMaskName(activeSubMask) })
            : t('editor.masks.settings.maskPropertiesTitle')
        }
        isOpen={isSettingsSectionOpen}
        onToggle={() => {
          const isOpening = !isSettingsSectionOpen;
          setSettingsSectionOpen(isOpening);
          if (appSettings?.enableFocusMode && isOpening) {
            setCollapsibleState((prev: any) => {
              const newState = { ...prev };
              Object.keys(newState).forEach((key) => {
                newState[key] = false;
              });
              return newState;
            });
          }
        }}
        canToggleVisibility={false}
        isContentVisible={true}
      >
        <div className="space-y-4 pt-2">
          <Switch
            checked={!!(isComponentMode ? activeSubMask.invert : displayContainer.invert)}
            label={isComponentMode ? t('editor.masks.settings.invertComponent') : t('editor.masks.settings.invertMask')}
            onChange={(v) =>
              isComponentMode ? updateSubMask(activeSubMask.id, { invert: v }) : handleMaskPropertyChange('invert', v)
            }
          />

          {!isComponentMode && (
            <div className="flex justify-between items-center">
              <Text variant={TextVariants.label} className="select-none">
                {t('editor.masks.settings.applyPreset')}
              </Text>
              <button
                ref={presetButtonRef}
                onClick={handlePresetSelectClick}
                className="ui-button ui-button--secondary ui-button--sm"
                data-tooltip={t('editor.masks.settings.selectPresetTooltip')}
              >
                {t('editor.masks.settings.select')}
              </button>
            </div>
          )}

          <Slider
            defaultValue={100}
            label={t('editor.masks.settings.opacity')}
            max={100}
            min={0}
            value={(isComponentMode ? activeSubMask.opacity : displayContainer.opacity) ?? 100}
            onChange={(e: any) =>
              isComponentMode
                ? updateSubMask(activeSubMask.id, { opacity: Number(e.target.value) })
                : handleMaskPropertyChange('opacity', Number(e.target.value))
            }
            step={1}
            fillOrigin="min"
            onDragStateChange={onDragStateChange}
          />

          {isComponentMode && (
            <>
              {isAiMask && (aiModelDownloadStatus || isGeneratingAiMask) && (
                <TaskProgress
                  ariaLabel={
                    aiModelDownloadStatus
                      ? t('editor.masks.settings.aiModelDownloading')
                      : t('editor.ai.settings.generating')
                  }
                  compact
                  indeterminate
                  label={
                    aiModelDownloadStatus
                      ? `${t('editor.masks.settings.aiModelDownloading')}: ${aiModelDownloadStatus}`
                      : t('editor.ai.settings.generating')
                  }
                />
              )}

              {activeSubMask.type === Mask.AiDepth && (
                <DepthRangePicker
                  minDepth={100 - (activeSubMask.parameters?.maxDepth ?? 100)}
                  maxDepth={100 - (activeSubMask.parameters?.minDepth ?? 0)}
                  minFade={activeSubMask.parameters?.maxFade ?? 15}
                  maxFade={activeSubMask.parameters?.minFade ?? 15}
                  defaultMinDepth={20}
                  defaultMaxDepth={80}
                  defaultMinFade={15}
                  defaultMaxFade={15}
                  onChange={handleDepthRangeChange}
                  onDragStateChange={onDragStateChange}
                />
              )}

              {subMaskConfig.parameters?.map((param: any) => (
                <Slider
                  key={param.key}
                  label={
                    param.key === 'feather' && activeSubMask.type === Mask.AiDepth
                      ? t('editor.masks.params.globalFeather')
                      : (t(('editor.masks.params.' + param.key) as never) as string)
                  }
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  defaultValue={param.defaultValue}
                  value={(activeSubMask.parameters[param.key] || 0) * (param.multiplier || 1)}
                  onChange={(e: any) =>
                    handleSubMaskParametersChange({ [param.key]: parseFloat(e.target.value) / (param.multiplier || 1) })
                  }
                  {...(param.key !== 'grow' && { fillOrigin: 'min' })}
                  onDragStateChange={onDragStateChange}
                />
              ))}

              {subMaskConfig.showBrushTools &&
                brushSettings &&
                (activeSubMask.type === Mask.Flow ? (
                  <FlowBrushTool
                    flow={activeSubMask.parameters?.flow ?? 10}
                    onFlowChange={(flow: number) => handleSubMaskParametersChange({ flow })}
                    settings={brushSettings}
                    onSettingsChange={setBrushSettings}
                    onDragStateChange={onDragStateChange}
                  />
                ) : (
                  <BrushTools
                    settings={brushSettings}
                    onSettingsChange={setBrushSettings}
                    onDragStateChange={onDragStateChange}
                  />
                ))}
            </>
          )}
        </div>
      </CollapsibleSection>

      <div
        onMouseEnter={() => setIsMaskControlHovered(true)}
        onMouseLeave={() => setIsMaskControlHovered(false)}
        className="flex flex-col gap-2"
      >
        {Object.keys(ADJUSTMENT_SECTIONS).map((sectionName) => {
          const SectionComponent: any = {
            basic: BasicAdjustments,
            curves: CurveGraph,
            color: ColorPanel,
            details: DetailsPanel,
            effects: EffectsPanel,
          }[sectionName];
          const title = sectionName.charAt(0).toUpperCase() + sectionName.slice(1);
          return (
            <CollapsibleSection
              key={sectionName}
              title={title}
              isOpen={collapsibleState[sectionName]}
              isContentVisible={sectionVisibility[sectionName]}
              onToggle={() => handleToggleSection(sectionName)}
              onToggleVisibility={() => handleToggleVisibility(sectionName)}
              onContextMenu={(e: any) => handleSectionContextMenu(e, sectionName)}
            >
              <SectionComponent
                adjustments={displayContainer.adjustments}
                setAdjustments={setMaskContainerAdjustments}
                histogram={histogram}
                isForMask={true}
                appSettings={appSettings}
                onDragStateChange={onDragStateChange}
              />
            </CollapsibleSection>
          );
        })}
      </div>
    </div>
  );
}
