import { type ChangeEvent } from 'react';
import { useTranslation } from 'react-i18next';

import Slider from '../../../ui/Slider';

import { ToolType } from '../Masks';

export const BrushTools = ({
  settings,
  onSettingsChange,
  onDragStateChange,
}: {
  settings: any;
  onSettingsChange: any;
  onDragStateChange?: (isDragging: boolean) => void;
}) => {
  const { t } = useTranslation();

  return (
    <div>
      <Slider
        defaultValue={100}
        label={t('editor.masks.brush.size')}
        max={200}
        min={1}
        onChange={(e: any) => onSettingsChange((s: any) => ({ ...s, size: Number(e.target.value) }))}
        step={1}
        value={settings.size}
        fillOrigin="min"
        onDragStateChange={onDragStateChange}
      />
      <Slider
        defaultValue={50}
        label={t('editor.masks.brush.feather')}
        max={100}
        min={0}
        onChange={(e: any) => onSettingsChange((s: any) => ({ ...s, feather: Number(e.target.value) }))}
        step={1}
        value={settings.feather}
        fillOrigin="min"
        onDragStateChange={onDragStateChange}
      />
      <div className="grid grid-cols-2 gap-2 pt-2">
        <button
          aria-pressed={settings.tool === ToolType.Brush}
          className={`ui-choice-button ${settings.tool === ToolType.Brush ? 'is-active' : ''}`}
          onClick={() => onSettingsChange((s: any) => ({ ...s, tool: ToolType.Brush }))}
          type="button"
        >
          {t('editor.masks.brush.brush')}
        </button>
        <button
          aria-pressed={settings.tool === ToolType.Eraser}
          className={`ui-choice-button ${settings.tool === ToolType.Eraser ? 'is-active' : ''}`}
          onClick={() => onSettingsChange((s: any) => ({ ...s, tool: ToolType.Eraser }))}
          type="button"
        >
          {t('editor.masks.brush.eraser')}
        </button>
      </div>
    </div>
  );
};

export const FlowBrushTool = ({
  flow,
  onFlowChange,
  settings,
  onSettingsChange,
  onDragStateChange,
}: {
  flow: number;
  onFlowChange: (flow: number) => void;
  settings: any;
  onSettingsChange: any;
  onDragStateChange?: (isDragging: boolean) => void;
}) => {
  const { t } = useTranslation();

  return (
    <div className="space-y-4 border-t border-border-color">
      <Slider
        defaultValue={10}
        label={t('editor.masks.brush.flow')}
        max={100}
        min={0}
        onChange={(e: ChangeEvent<HTMLInputElement>) => onFlowChange(Number(e.target.value))}
        step={1}
        value={flow}
        fillOrigin="min"
        onDragStateChange={onDragStateChange}
      />
      <BrushTools settings={settings} onSettingsChange={onSettingsChange} onDragStateChange={onDragStateChange} />
    </div>
  );
};
