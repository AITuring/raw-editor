export const DEFAULT_LANGUAGE = 'en';

export const SUPPORTED_LANGUAGES = [
  'en',
  'zh-CN',
  'zh-TW',
  'de',
  'es',
  'fr',
  'it',
  'ja',
  'ko',
  'pl',
  'pt',
  'ru',
] as const;

export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

export const LANGUAGE_OPTIONS: ReadonlyArray<{ value: SupportedLanguage; label: string }> = [
  { value: 'en', label: 'English' },
  { value: 'zh-CN', label: '简体中文' },
  { value: 'zh-TW', label: '繁體中文' },
  { value: 'de', label: 'Deutsch' },
  { value: 'es', label: 'Español' },
  { value: 'fr', label: 'Français' },
  { value: 'it', label: 'Italiano' },
  { value: 'ja', label: '日本語' },
  { value: 'ko', label: '한국어' },
  { value: 'pl', label: 'Polski' },
  { value: 'pt', label: 'Português' },
  { value: 'ru', label: 'Русский' },
];

const findSupportedLanguage = (language?: string | null): SupportedLanguage | undefined => {
  if (!language) return undefined;

  const normalized = language.trim().replaceAll('_', '-').toLowerCase();
  const exactMatch = SUPPORTED_LANGUAGES.find((candidate) => candidate.toLowerCase() === normalized);
  if (exactMatch) return exactMatch;

  if (
    normalized.startsWith('zh-hant') ||
    normalized.startsWith('zh-tw') ||
    normalized.startsWith('zh-hk') ||
    normalized.startsWith('zh-mo')
  ) {
    return 'zh-TW';
  }
  if (normalized === 'zh' || normalized.startsWith('zh-')) {
    return 'zh-CN';
  }

  const baseLanguage = normalized.split('-')[0];
  return SUPPORTED_LANGUAGES.find((candidate) => candidate.toLowerCase() === baseLanguage);
};

export const resolveSupportedLanguage = (language?: string | null): SupportedLanguage =>
  findSupportedLanguage(language) ?? DEFAULT_LANGUAGE;

export const getSystemLanguage = (): SupportedLanguage => {
  if (typeof navigator === 'undefined') return DEFAULT_LANGUAGE;

  const candidates = [...(navigator.languages ?? []), navigator.language];
  for (const candidate of candidates) {
    const supportedLanguage = findSupportedLanguage(candidate);
    if (supportedLanguage) return supportedLanguage;
  }

  return DEFAULT_LANGUAGE;
};
