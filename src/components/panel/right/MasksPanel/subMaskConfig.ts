import { Mask } from '../Masks';

export const SUB_MASK_CONFIG: Record<Mask, any> = {
  [Mask.Radial]: {
    parameters: [{ key: 'feather', min: 0, max: 100, step: 1, multiplier: 100, defaultValue: 50 }],
  },
  [Mask.Brush]: { showBrushTools: true },
  [Mask.Flow]: { showBrushTools: true, showFlowControl: true },
  [Mask.Clone]: { showBrushTools: true },
  [Mask.Heal]: { showBrushTools: true },
  [Mask.Linear]: { parameters: [] },
  [Mask.Color]: {
    parameters: [
      { key: 'tolerance', min: 1, max: 100, step: 1, defaultValue: 20 },
      { key: 'grow', min: -100, max: 100, step: 1, defaultValue: 0 },
      { key: 'feather', min: 0, max: 100, step: 1, defaultValue: 35 },
    ],
  },
  [Mask.Luminance]: {
    parameters: [
      { key: 'tolerance', min: 1, max: 100, step: 1, defaultValue: 20 },
      { key: 'grow', min: -100, max: 100, step: 1, defaultValue: 0 },
      { key: 'feather', min: 0, max: 100, step: 1, defaultValue: 35 },
    ],
  },
  [Mask.All]: { parameters: [] },
  [Mask.AiDepth]: {
    parameters: [{ key: 'feather', min: 0, max: 100, step: 1, defaultValue: 15 }],
  },
  [Mask.AiSubject]: {
    parameters: [
      { key: 'grow', min: -100, max: 100, step: 1, defaultValue: 0 },
      { key: 'feather', min: 0, max: 100, step: 1, defaultValue: 0 },
    ],
  },
  [Mask.AiForeground]: {
    parameters: [
      { key: 'grow', min: -100, max: 100, step: 1, defaultValue: 0 },
      { key: 'feather', min: 0, max: 100, step: 1, defaultValue: 0 },
    ],
  },
  [Mask.AiSky]: {
    parameters: [
      { key: 'grow', min: -100, max: 100, step: 1, defaultValue: 0 },
      { key: 'feather', min: 0, max: 100, step: 1, defaultValue: 0 },
    ],
  },
  [Mask.QuickEraser]: { parameters: [] },
};
