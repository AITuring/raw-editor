import { useTranslation } from 'react-i18next';

import Slider from '../../../ui/Slider';

import { ToolType } from '../Masks';

export const BrushTools = ({ settings, onSettingsChange }: { settings: any; onSettingsChange: any }) => {
  const { t } = useTranslation();

  return (
    <div>
      <Slider
        defaultValue={100}
        label={t('editor.ai.brush.size')}
        max={200}
        min={1}
        onChange={(e: any) => onSettingsChange((s: any) => ({ ...s, size: Number(e.target.value) }))}
        step={1}
        value={settings.size}
        fillOrigin="min"
      />
      <Slider
        defaultValue={50}
        label={t('editor.ai.brush.feather')}
        max={100}
        min={0}
        onChange={(e: any) => onSettingsChange((s: any) => ({ ...s, feather: Number(e.target.value) }))}
        step={1}
        value={settings.feather}
        fillOrigin="min"
      />
      <div className="grid grid-cols-2 gap-2 pt-2">
        <button
          aria-pressed={settings.tool === ToolType.Brush}
          className={`ui-choice-button ${settings.tool === ToolType.Brush ? 'is-active' : ''}`}
          onClick={() => onSettingsChange((s: any) => ({ ...s, tool: ToolType.Brush }))}
          type="button"
        >
          {t('editor.ai.brush.add')}
        </button>
        <button
          aria-pressed={settings.tool === ToolType.Eraser}
          className={`ui-choice-button ${settings.tool === ToolType.Eraser ? 'is-active' : ''}`}
          onClick={() => onSettingsChange((s: any) => ({ ...s, tool: ToolType.Eraser }))}
          type="button"
        >
          {t('editor.ai.brush.erase')}
        </button>
      </div>
    </div>
  );
};
