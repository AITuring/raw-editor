import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { getSystemLanguage, SUPPORTED_LANGUAGES } from './languages';

import en from './locales/en.json';
import de from './locales/de.json';
import zhCN from './locales/zh-CN.json';
import zhTW from './locales/zh-TW.json';
import pl from './locales/pl.json';
import es from './locales/es.json';
import fr from './locales/fr.json';
import it from './locales/it.json';
import pt from './locales/pt.json';
import ja from './locales/ja.json';
import ko from './locales/ko.json';
import ru from './locales/ru.json';

const initialLanguage = getSystemLanguage();

void i18n.use(initReactI18next).init({
  resources: {
    en: { translation: en },
    de: { translation: de },
    'zh-CN': { translation: zhCN },
    'zh-TW': { translation: zhTW },
    pl: { translation: pl },
    es: { translation: es },
    fr: { translation: fr },
    it: { translation: it },
    pt: { translation: pt },
    ja: { translation: ja },
    ko: { translation: ko },
    ru: { translation: ru },
  },
  lng: initialLanguage,
  fallbackLng: 'en',
  supportedLngs: [...SUPPORTED_LANGUAGES],
  load: 'currentOnly',
  returnEmptyString: false,
  interpolation: {
    escapeValue: false,
  },
});

const updateDocumentLanguage = (language: string) => {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = language;
  document.documentElement.dir = i18n.dir(language);
};

updateDocumentLanguage(initialLanguage);
i18n.on('languageChanged', updateDocumentLanguage);

export default i18n;
