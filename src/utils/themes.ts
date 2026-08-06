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
      '--app-status-info': 'rgb(112, 176, 224)',
      '--app-status-success': 'rgb(111, 190, 137)',
      '--app-status-warning': 'rgb(230, 178, 90)',
      '--app-status-error': 'rgb(231, 117, 117)',
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
      '--app-status-info': 'rgb(31, 104, 167)',
      '--app-status-success': 'rgb(35, 122, 69)',
      '--app-status-warning': 'rgb(145, 91, 17)',
      '--app-status-error': 'rgb(177, 54, 54)',
      '--app-border-color': 'rgb(224, 224, 224)',
      '--app-hover-color': 'rgb(198, 142, 110)',
    },
  },
  {
    id: Theme.Grey,
    name: 'settings.themes.grey',
    cssVariables: {
      '--app-bg-primary': 'rgb(48, 49, 50)',
      '--app-bg-secondary': 'rgb(75, 76, 77)',
      '--app-surface': 'rgb(68, 69, 70)',
      '--app-card-active': 'rgb(88, 89, 90)',
      '--app-button-text': 'rgb(45, 42, 39)',
      '--app-text-primary': 'rgb(238, 237, 234)',
      '--app-text-secondary': 'rgb(187, 186, 182)',
      '--app-accent': 'rgb(222, 169, 119)',
      '--app-status-info': 'rgb(137, 196, 238)',
      '--app-status-success': 'rgb(130, 204, 153)',
      '--app-status-warning': 'rgb(239, 191, 112)',
      '--app-status-error': 'rgb(239, 137, 137)',
      '--app-border-color': 'rgb(50, 51, 52)',
      '--app-hover-color': 'rgb(222, 169, 119)',
    },
  },
];

export const DEFAULT_THEME_ID = Theme.Dark;
