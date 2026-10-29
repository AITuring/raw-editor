import { OptionItem } from '../../ui/Dropdown';

export const THIRD_PARTY_DISPLAY_NAMES = {
  rawler: 'rawler',
  lensfun: 'lensfun',
  negpy: 'NegPy',
  lama: 'LaMa',
  sam2: 'SAM 2',
  u2net: 'U-2-Net',
  depthAnything: 'Depth Anything V2',
  nind: 'nind-denoise',
  darktable: 'darktable & co.',
} as const;

export const EXECUTE_TIMEOUT = 3000;

export const adjustmentVisibilityDefaults = {
  sharpening: true,
  presence: true,
  noiseReduction: true,
  chromaticAberration: false,
  vignette: true,
  colorCalibration: false,
  grain: true,
};

export const resolutions: OptionItem<number>[] = [
  { value: 720, label: '720px' },
  { value: 1280, label: '1280px' },
  { value: 1920, label: '1920px' },
  { value: 2560, label: '2560px' },
  { value: 3840, label: '3840px' },
];

export const thumbnailResolutions: OptionItem<number>[] = [
  { value: 640, label: '640px' },
  { value: 720, label: '720px' },
  { value: 960, label: '960px' },
  { value: 1080, label: '1080px' },
];

export const zoomMultiplierOptions: OptionItem<number>[] = [
  { value: 1.0, label: '1.0x (Native)' },
  { value: 0.75, label: '0.75x' },
  { value: 0.5, label: '0.50x (Half)' },
  { value: 0.25, label: '0.25x' },
];
