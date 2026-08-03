import { defineConfig } from 'i18next-cli';
import { SUPPORTED_LANGUAGES } from './src/i18n/languages';

export default defineConfig({
  locales: [...SUPPORTED_LANGUAGES],
  extract: {
    input: ['src/**/*.{ts,tsx}'],
    output: 'src/i18n/locales/{{language}}.json',
    defaultNS: false,
    removeUnusedKeys: false,
    sort: true,
    defaultValue: '',
  },
});
