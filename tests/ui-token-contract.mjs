import { readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';

const root = resolve(import.meta.dirname, '..');
const read = (file) => readFileSync(resolve(root, file), 'utf8');
const tokens = read('src/styles/tokens.css');
const styles = read('src/styles.css');
const failures = [];

const requiredTokens = [
  '--ui-panel-inset',
  '--ui-editor-panel-inset',
  '--ui-floating-viewport-inset',
  '--ui-dialog-inset',
  '--ui-dialog-footer-gap',
  '--ui-dialog-footer-padding-block',
  '--ui-denoise-control-column',
  '--ui-denoise-preview-height',
  '--ui-denoise-result-height',
  '--ui-panel-header-height',
  '--ui-toolbar-height',
  '--ui-statusbar-height',
  '--ui-tool-rail-width',
  '--ui-editor-adjustment-inset',
  '--ui-editor-section-leading-column',
  '--ui-editor-section-header-height',
  '--ui-editor-subsection-header-height',
  '--ui-editor-field-label-column',
  '--ui-editor-value-column',
  '--ui-editor-control-gap',
  '--ui-editor-section-padding-block',
  '--ui-editor-slider-track-height',
  '--ui-editor-slider-row-gap',
  '--ui-editor-action-min-height',
  '--ui-editor-header-action-min-width',
  '--ui-editor-description-measure',
  '--ui-editor-scrollbar-reserve',
  '--ui-border-control',
  '--ui-border-structural',
  '--ui-border-floating',
  '--ui-button-height',
  '--ui-button-height-sm',
  '--ui-shadow-button-rest',
  '--ui-shadow-button-hover',
  '--ui-shadow-button-pressed',
  '--ui-shadow-button-selected',
  '--ui-shadow-tool-rest',
  '--ui-shadow-tool-hover',
  '--ui-shadow-tool-selected',
  '--ui-shadow-message',
  '--ui-surface-tool-rest',
  '--ui-surface-tool-hover',
  '--ui-surface-tool-selected',
  '--ui-filter-popover-width',
  '--ui-filter-popover-trigger-gap',
  '--ui-copy-paste-dialog-width',
  '--ui-copy-paste-dialog-max-height',
  '--ui-denoise-dialog-max-width',
  '--ui-message-max-width',
  '--ui-message-min-height',
  '--ui-message-icon-size',
  '--ui-message-gap',
  '--ui-message-padding-block',
  '--ui-message-padding-inline',
];

for (const token of requiredTokens) {
  if (!tokens.includes(`${token}:`)) failures.push(`Missing required token: ${token}`);
}

for (const token of ['--ui-shadow-button-selected', '--ui-shadow-tool-selected']) {
  if (!tokens.includes(`${token}:`) || !tokens.includes('inset 0 0 0 1px var(--app-accent)')) {
    failures.push(`${token} must use a four-edge inset selection highlight.`);
  }
}

for (const token of ['--ui-shadow-tool-rest', '--ui-shadow-tool-hover']) {
  if (!tokens.includes(`${token}: none;`)) failures.push(`${token} must keep contextual tool buttons flat.`);
}

if (tokens.includes('inset 0 -2px 0 var(--app-accent)')) {
  failures.push('Selected controls must not use a single-edge underline highlight.');
}

if (styles.indexOf("@import './styles/tokens.css';") > styles.indexOf("@import 'tailwindcss';")) {
  failures.push('tokens.css must load before Tailwind.');
}

const selectorContracts = [
  ['.ui-panel-header', '--ui-panel-inset'],
  ['.ui-panel-body', '--ui-panel-inset'],
  ['.ui-toolbar', '--ui-toolbar-height'],
  ['.ui-icon-button', '--ui-size-icon-hit'],
  ['.ui-icon-button', '--ui-shadow-tool-rest'],
  ['.develop-tool-button', '--ui-shadow-tool-rest'],
  ['.ui-button', '--ui-button-height'],
  ['.ui-segmented-option', '--ui-button-height-sm'],
  ['.editor-status-bar', '--ui-statusbar-height'],
  ['.editor-filter-popover', '--ui-filter-popover-width'],
  ['.develop-tool-rail', '--develop-tool-rail-size'],
  ['.develop-panel-content :is(.ui-panel-header', '--ui-editor-panel-inset'],
  ['.develop-controls-panel .develop-collapsible-header', '--ui-editor-section-header-height'],
  ['.develop-controls-panel > .ui-panel-header .ui-button', '--ui-editor-header-action-min-width'],
  ['.develop-controls-panel .camera-raw-slider-header', '--ui-editor-value-column'],
  ['.develop-controls-panel .camera-raw-action-row', '--ui-editor-action-min-height'],
  ['.develop-controls-panel .camera-raw-select', '--ui-shadow-button-rest'],
  ['.app-message {', '--ui-shadow-message'],
  ['.app-modal-surface--padded', '--ui-dialog-inset'],
  ['.app-modal-footer', '--ui-dialog-footer-gap'],
  ['.denoise-modal-surface', '--ui-denoise-dialog-max-width'],
  ['.denoise-modal-body', '--ui-dialog-inset'],
  ['.denoise-modal-footer', '--ui-dialog-inset'],
];

for (const [selector, token] of selectorContracts) {
  const selectorIndex = styles.indexOf(selector);
  const tokenIndex = styles.indexOf(token, selectorIndex);
  if (selectorIndex < 0 || tokenIndex < 0 || tokenIndex - selectorIndex > 900) {
    failures.push(`${selector} must consume ${token}.`);
  }
}

const denoiseSurfaceRule = styles.match(/\.denoise-modal-surface\s*\{([^}]*)\}/s)?.[1] ?? '';
if (!denoiseSurfaceRule.includes('width: min(100%, var(--ui-denoise-dialog-max-width));')) {
  failures.push('Denoise modal must keep a tokenized width cap instead of expanding to the viewport.');
}
if (!styles.includes('.denoise-modal-footer.app-modal-footer--inset')) {
  failures.push('Denoise modal footer must explicitly neutralize nested inset padding when using margin inset.');
}

if (!styles.includes('button {\n  border: 0;\n}')) {
  failures.push('The global button boundary contract must keep every button borderless.');
}

if (!styles.includes('box-shadow: var(--ui-shadow-button-rest);')) {
  failures.push('Raised buttons must consume --ui-shadow-button-rest.');
}

const messageRuleStart = styles.indexOf('.app-message {');
const messageRuleEnd = styles.indexOf('}', messageRuleStart);
const messageRule =
  messageRuleStart >= 0 && messageRuleEnd > messageRuleStart ? styles.slice(messageRuleStart, messageRuleEnd) : '';
if (!messageRule.includes('border: 0;')) {
  failures.push('Transient messages must remain borderless.');
}

if (!styles.includes('.ui-button:is(.ui-button--ghost, .ui-button--icon)')) {
  failures.push('Ghost and icon Button variants must share the flat tool-button contract.');
}

if (!styles.includes('.ui-tab-action.is-active {') || !styles.includes('box-shadow: var(--ui-shadow-tool-selected);')) {
  failures.push('Active tabs must use the shared four-edge selection highlight.');
}

if (
  styles.includes('.develop-tool-indicator') ||
  read('src/components/panel/DevelopPanel.tsx').includes('develop-tool-indicator')
) {
  failures.push('Develop tool buttons must not reintroduce a single-edge active indicator.');
}

const controlsPanelSource = read('src/components/panel/right/ControlsPanel.tsx');
const adjustmentsSource = read('src/utils/adjustments.ts');
if (!controlsPanelSource.includes('variant="primary"')) {
  failures.push('Develop auto-adjust must remain a raised primary action.');
}
if (!controlsPanelSource.includes('variant="secondary"')) {
  failures.push('Develop reset must remain a raised secondary action.');
}
if (!controlsPanelSource.includes('canToggleVisibility={false}')) {
  failures.push('Develop adjustment sections must not expose visibility toggles.');
}
if (controlsPanelSource.includes('onToggleVisibility=')) {
  failures.push('Develop adjustment sections must not wire the removed eye action.');
}
if (
  !adjustmentsSource.includes(
    'const normalizedSectionVisibility: SectionVisibility = { ...INITIAL_ADJUSTMENTS.sectionVisibility };',
  )
) {
  failures.push('Legacy global section visibility must normalize to the always-active Develop pipeline.');
}

const flatToolSelectors = [
  '.develop-tool-button',
  '.camera-raw-wb-picker',
  '.develop-panel-content button.camera-raw-slider-number',
  '.crop-icon-button',
  '.editor-rating-group',
  '.editor-library-action',
  '.ui-button:is(.ui-button--ghost, .ui-button--icon)',
];

for (const selector of flatToolSelectors) {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const rule = styles.match(new RegExp(`${escapedSelector}\\s*\\{([^}]*)\\}`, 's'))?.[1] ?? '';
  if (!rule.includes('box-shadow: var(--ui-shadow-tool-rest);')) {
    failures.push(`${selector} must use the flat tool-button elevation token.`);
  }
}

const walkFiles = (directory) =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = resolve(directory, entry.name);
    return entry.isDirectory() ? walkFiles(path) : [path];
  });

const sourceComponentFiles = walkFiles(resolve(root, 'src')).filter((path) => path.endsWith('.tsx'));

for (const file of sourceComponentFiles) {
  const source = readFileSync(file, 'utf8');

  if (
    source.includes('app-modal-surface') &&
    !['app-modal-surface--padded', 'app-modal-surface--structured', 'app-modal-surface--full-bleed'].some((variant) =>
      source.includes(variant),
    )
  ) {
    failures.push(`${file} must declare an explicit app-modal-surface layout variant.`);
  }

  const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

  const visit = (node) => {
    if (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node)) {
      const tagName = node.tagName.getText(sourceFile);
      if (tagName === 'button' || tagName === 'Button') {
        const className = node.attributes.properties.find(
          (property) => ts.isJsxAttribute(property) && property.name.getText(sourceFile) === 'className',
        );

        if (className && /\bborder(?:-|\b)/.test(className.getText(sourceFile))) {
          const { line } = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile));
          failures.push(`${file}:${line + 1} adds a border utility to a button.`);
        }
      }
    }

    ts.forEachChild(node, visit);
  };

  visit(sourceFile);
}

const componentContracts = [
  ['src/components/panel/editor/EditorToolbar.tsx', 'ui-toolbar editor-command-bar'],
  ['src/components/panel/MainLibrary.tsx', 'ui-toolbar ui-library-toolbar'],
  ['src/components/panel/MainLibrary.tsx', 'ui-chrome-panel'],
  ['src/components/panel/BottomBar.tsx', 'editor-bottom-dock'],
  ['src/components/panel/LibraryNavigationPanel.tsx', 'ui-chrome-panel'],
  ['src/components/panel/LibraryContextPanel.tsx', 'ui-chrome-panel'],
  ['src/components/panel/right/ControlsPanel.tsx', 'ui-panel-root'],
  ['src/components/panel/right/ControlsPanel.tsx', 'develop-controls-panel'],
  ['src/components/panel/right/CropPanel.tsx', 'ui-panel-root'],
  ['src/components/panel/right/MasksPanel.tsx', 'ui-panel-root'],
  ['src/components/panel/right/AIPanel.tsx', 'ui-panel-root'],
  ['src/components/panel/right/PresetsPanel.tsx', 'ui-panel-root'],
  ['src/components/panel/right/MetadataPanel.tsx', 'ui-panel-root'],
  ['src/components/panel/right/ExportPanel.tsx', 'ui-panel-root'],
  ['src/components/ui/CollapsibleSection.tsx', 'develop-collapsible-content'],
  ['src/components/ui/CollapsibleSection.tsx', "isOpen && 'is-open'"],
  ['src/components/modals/CopyPasteSettingsModal.tsx', 'copy-paste-dialog'],
  ['src/components/modals/CopyPasteSettingsModal.tsx', 'app-modal-surface--structured'],
  ['src/components/modals/DenoiseModal.tsx', 'denoise-modal-surface'],
  ['src/components/modals/DenoiseModal.tsx', 'app-modal-surface--structured'],
  ['src/components/modals/ConfigurePresetModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/ConfirmModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/CreateFolderModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/CullingModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/HdrModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/ImportSettingsModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/PanoramaModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/RenameFileModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/RenameFolderModal.tsx', 'app-modal-surface--padded'],
  ['src/components/modals/CollageModal.tsx', 'app-modal-surface--full-bleed'],
  ['src/components/modals/LensCorrectionModal.tsx', 'app-modal-surface--full-bleed'],
  ['src/components/modals/NegativeConversionModal.tsx', 'app-modal-surface--full-bleed'],
  ['src/components/modals/TransformModal.tsx', 'app-modal-surface--full-bleed'],
  ['src/features/export/ExportImageDialog.tsx', 'app-modal-surface--full-bleed'],
];

for (const [file, className] of componentContracts) {
  if (!read(file).includes(className)) failures.push(`${file} must use ${className}.`);
}

if (read('src/components/panel/DevelopPanel.tsx').includes('DEVELOP_TOOL_RAIL_WIDTH')) {
  failures.push('DevelopPanel must use --ui-tool-rail-width instead of a local width constant.');
}

if (read('src/components/panel/BottomBar.tsx').includes('QUICK_FILTER_WIDTH')) {
  failures.push('BottomBar must use --ui-filter-popover-width instead of a local width constant.');
}

if (!read('src/components/panel/BottomBar.tsx').includes('--ui-floating-viewport-inset')) {
  failures.push('BottomBar popovers must use --ui-floating-viewport-inset for viewport edge safety.');
}

if (!read('src/components/panel/BottomBar.tsx').includes('--ui-filter-popover-trigger-gap')) {
  failures.push('BottomBar quick filter must use --ui-filter-popover-trigger-gap for trigger separation.');
}

const legacyOversizedButtonPatterns = [
  'px-4 py-2 rounded-md text-text-secondary',
  'relative flex-1 flex items-center justify-center gap-2 px-3 py-1.5 text-sm font-medium',
  'p-2 rounded-md text-sm font-medium transition-colors flex items-center',
];

const buttonMigrationFiles = [
  'src/components/adjustments/Effects.tsx',
  'src/components/modals/CollageModal.tsx',
  'src/components/modals/ConfigurePresetModal.tsx',
  'src/components/modals/CullingModal.tsx',
  'src/components/modals/DenoiseModal.tsx',
  'src/components/modals/HdrModal.tsx',
  'src/components/modals/ImageStackModal.tsx',
  'src/components/modals/LensCorrectionModal.tsx',
  'src/components/modals/NegativeConversionModal.tsx',
  'src/components/modals/PanoramaModal.tsx',
  'src/components/modals/TransformModal.tsx',
  'src/components/panel/SettingsPanel.tsx',
  'src/components/panel/right/AIPanel.tsx',
  'src/components/panel/right/ControlsPanel.tsx',
  'src/components/panel/right/MasksPanel.tsx',
];

for (const file of buttonMigrationFiles) {
  const source = read(file);
  for (const pattern of legacyOversizedButtonPatterns) {
    if (source.includes(pattern)) failures.push(`${file} still contains legacy oversized button styling: ${pattern}`);
  }
}

if (failures.length > 0) {
  console.error(failures.map((failure) => `- ${failure}`).join('\n'));
  process.exit(1);
}

console.log(`UI token contract passed (${requiredTokens.length} tokens, ${componentContracts.length} consumers).`);
