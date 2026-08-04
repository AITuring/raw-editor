import { Theme } from '../components/ui/AppProperties';

export interface ThemeProps {
  cssVariables: any;
  id: Theme;
  name: string;
}

export const THEMES: Array<ThemeProps> = [
  {
    id: Theme.Dark,
    name: 'settings.themes.dark',
    cssVariables: {
      '--app-bg-primary': 'rgb(21, 22, 23)',
      '--app-bg-secondary': 'rgb(30, 31, 32)',
      '--app-surface': 'rgb(36, 37, 38)',
      '--app-card-active': 'rgb(45, 46, 48)',
      '--app-button-text': 'rgb(29, 25, 21)',
      '--app-text-primary': 'rgb(232, 231, 228)',
      '--app-text-secondary': 'rgb(158, 157, 153)',
      '--app-accent': 'rgb(211, 157, 106)',
      '--app-border-color': 'rgb(51, 52, 54)',
      '--app-hover-color': 'rgb(211, 157, 106)',
    },
  },
  {
    id: Theme.Light,
    name: 'settings.themes.light',
    cssVariables: {
      '--app-bg-primary': 'rgb(245, 245, 245)',
      '--app-bg-secondary': 'rgb(255, 255, 255)',
      '--app-surface': 'rgb(241, 241, 241)',
      '--app-card-active': 'rgb(250, 250, 250)',
      '--app-button-text': 'rgb(255, 255, 255)',
      '--app-text-primary': 'rgb(20, 20, 20)',
      '--app-text-secondary': 'rgb(108, 108, 108)',
      '--app-accent': 'rgb(198, 142, 110)',
      '--app-border-color': 'rgb(224, 224, 224)',
      '--app-hover-color': 'rgb(198, 142, 110)',
    },
  },
  {
    id: Theme.Grey,
    name: 'settings.themes.grey',
    cssVariables: {
      '--app-bg-primary': 'rgb(112, 112, 112)',
      '--app-bg-secondary': 'rgb(118, 118, 118)',
      '--app-surface': 'rgb(108, 108, 108)',
      '--app-card-active': 'rgb(133, 133, 133)',
      '--app-button-text': 'rgb(45, 45, 45)',
      '--app-text-primary': 'rgb(240, 240, 240)',
      '--app-text-secondary': 'rgb(180, 180, 180)',
      '--app-accent': 'rgb(220, 220, 220)',
      '--app-border-color': 'rgb(138, 138, 138)',
      '--app-hover-color': 'rgb(220, 220, 220)',
    },
  },
];

export const DEFAULT_THEME_ID = Theme.Dark;
