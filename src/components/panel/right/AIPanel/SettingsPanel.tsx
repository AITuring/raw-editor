import { useTranslation } from 'react-i18next';
import { Loader2, Send } from 'lucide-react';

import CollapsibleSection from '../../../ui/CollapsibleSection';
import Switch from '../../../ui/Switch';
import Slider from '../../../ui/Slider';
import Button from '../../../ui/Button';
import TaskProgress from '../../../ui/TaskProgress';

import { Mask, SubMask, getSubMaskName } from '../Masks';
import { AiPatch } from '../../../../utils/adjustments';
import Text from '../../../ui/Text';
import { TextVariants } from '../../../../types/typography';
import { BrushTools } from './BrushTools';
import { SUB_MASK_CONFIG } from './subMaskConfig';

const PLACEHOLDER_PATCH: AiPatch = {
  id: 'placeholder',
  invert: false,
  isLoading: false,
  name: '',
  prompt: '',
  subMasks: [],
  visible: true,
  patchData: null,
};

export function SettingsPanel({
  container,
  activeSubMask,
  aiModelDownloadStatus,
  brushSettings,
  setBrushSettings,
  updateContainer,
  updateSubMask,
  isGeneratingAi,
  isGeneratingAiMask,
  onGenerativeReplace,
  collapsibleState,
  setCollapsibleState,
}: any) {
  const { t } = useTranslation();
  const isActive = !!container;
  const isComponentMode = !!activeSubMask;
  const displayContainer = container || PLACEHOLDER_PATCH;
  const isQuickErasePatch = displayContainer.subMasks?.some((sm: SubMask) => sm.type === Mask.QuickEraser);
  const isCloneOrHealPatch = displayContainer.subMasks?.some(
    (sm: SubMask) => sm.type === Mask.Clone || sm.type === Mask.Heal,
  );
  const isStandalone =
    displayContainer?.subMasks?.length === 1 && [Mask.Clone, Mask.Heal].includes(displayContainer.subMasks[0].type);

  const subMaskConfig = activeSubMask ? SUB_MASK_CONFIG[activeSubMask.type] || {} : {};
  const isAiMask =
    activeSubMask &&
    (activeSubMask.type === Mask.AiSubject ||
      activeSubMask.type === Mask.AiForeground ||
      activeSubMask.type === Mask.AiSky);

  const handleGenerateClick = () => {
    if (!container) return;
    onGenerativeReplace(container.id, '', true);
  };

  const handleToggleSection = (section: string) =>
    setCollapsibleState((prev: any) => ({ ...prev, [section]: !prev[section] }));

  return (
    <div
      className={`space-y-2 transition-opacity duration-300 ${!isActive ? 'opacity-50 pointer-events-none' : ''}`}
      onClick={(e) => e.stopPropagation()}
    >
      {!isCloneOrHealPatch && (
        <CollapsibleSection
          title={t('editor.ai.settings.generativeReplaceTitle')}
          isOpen={collapsibleState.generative}
          onToggle={() => handleToggleSection('generative')}
          canToggleVisibility={false}
          isContentVisible={true}
        >
          <div className="space-y-4 pt-2">
            {aiModelDownloadStatus && aiModelDownloadStatus.includes('Inpainting') && (
              <TaskProgress
                ariaLabel={t('editor.ai.settings.downloading')}
                compact
                indeterminate
                label={`${t('editor.ai.settings.downloading')}: ${aiModelDownloadStatus}`}
              />
            )}

            <Text variant={TextVariants.small}>
              {isQuickErasePatch ? t('editor.ai.settings.quickEraseDesc') : t('editor.ai.settings.fastInpaintDesc')}
            </Text>

            <Button
              className="w-full"
              disabled={isGeneratingAi || displayContainer.isLoading || displayContainer.subMasks.length === 0}
              onClick={handleGenerateClick}
            >
              {isGeneratingAi || displayContainer.isLoading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Send size={16} />
              )}
              <span className="ml-2">
                {isGeneratingAi || displayContainer.isLoading
                  ? t('editor.ai.settings.generating')
                  : t('editor.ai.settings.inpaintSelectionButton')}
              </span>
            </Button>
            {(isGeneratingAi || displayContainer.isLoading) && (
              <TaskProgress
                ariaLabel={t('editor.ai.settings.generating')}
                compact
                indeterminate
                label={t('editor.ai.settings.generating')}
              />
            )}
          </div>
        </CollapsibleSection>
      )}

      <CollapsibleSection
        title={
          isStandalone
            ? t('editor.ai.settings.selectionPropertiesTitle')
            : isComponentMode
              ? t('editor.ai.settings.componentPropertiesTitle', { name: getSubMaskName(activeSubMask) })
              : t('editor.ai.settings.selectionPropertiesTitle')
        }
        isOpen={collapsibleState.properties}
        onToggle={() => handleToggleSection('properties')}
        canToggleVisibility={false}
        isContentVisible={true}
      >
        <div className="space-y-4 pt-2">
          {!isCloneOrHealPatch && (
            <Switch
              checked={!!(isComponentMode ? activeSubMask.invert : displayContainer.invert)}
              label={
                isComponentMode && !isStandalone
                  ? t('editor.ai.settings.invertComponent')
                  : t('editor.ai.settings.invertSelection')
              }
              onChange={(v) =>
                isComponentMode
                  ? updateSubMask(activeSubMask.id, { invert: v })
                  : updateContainer(container.id, { invert: v })
              }
            />
          )}

          {isComponentMode && (
            <>
              {isAiMask && (aiModelDownloadStatus || isGeneratingAiMask) && (
                <TaskProgress
                  ariaLabel={
                    aiModelDownloadStatus
                      ? t('editor.ai.settings.aiModelDownloading')
                      : t('editor.ai.settings.generating')
                  }
                  compact
                  indeterminate
                  label={
                    aiModelDownloadStatus
                      ? `${t('editor.ai.settings.aiModelDownloading')}: ${aiModelDownloadStatus}`
                      : t('editor.ai.settings.generating')
                  }
                />
              )}

              {subMaskConfig.parameters?.map((param: any) => (
                <Slider
                  key={param.key}
                  label={t(('editor.ai.params.' + param.key) as never) as string}
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  defaultValue={param.defaultValue}
                  value={(activeSubMask.parameters[param.key] || 0) * (param.multiplier || 1)}
                  onChange={(e: any) =>
                    updateSubMask(activeSubMask.id, {
                      parameters: {
                        ...activeSubMask.parameters,
                        [param.key]: parseFloat(e.target.value) / (param.multiplier || 1),
                      },
                    })
                  }
                  {...(param.key !== 'grow' && { fillOrigin: 'min' })}
                />
              ))}

              {subMaskConfig.showBrushTools && brushSettings && (
                <BrushTools settings={brushSettings} onSettingsChange={setBrushSettings} />
              )}
            </>
          )}
        </div>
      </CollapsibleSection>
    </div>
  );
}
