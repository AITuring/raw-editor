import type { ExportMetadataOverrides, ExportSettings } from '../../components/ui/ExportImportProperties';

export type ExportDialogFormat = 'jpeg' | 'png' | 'tiff';
export type ExportMetadataMode = 'none' | 'copyright' | 'all';

export interface ExportDialogSource {
  detailPreviewSrc?: string | null;
  fileName: string;
  height: number;
  previewSrc: string;
  width: number;
}

export interface ExportDialogSettings {
  artist: string;
  contact: string;
  copyright: string;
  description: string;
  embedColorProfile: boolean;
  format: ExportDialogFormat;
  jpegQuality: number;
  metadataEditedFields: Record<'artist' | 'contact' | 'copyright' | 'description', boolean>;
  metadataMode: ExportMetadataMode;
  resizeHeight: number;
  resizePercent: number;
  resizeWidth: number;
  sourceHeight: number;
  sourceWidth: number;
  stripGps: boolean;
}

export interface ExportDialogResult extends ExportDialogSettings {
  backendSettings: ExportSettings;
}

export const EXPORT_DIALOG_FORMATS: ReadonlyArray<{
  extensions: string[];
  id: ExportDialogFormat;
  label: string;
}> = [
  { id: 'jpeg', label: 'JPG', extensions: ['jpg', 'jpeg'] },
  { id: 'png', label: 'PNG', extensions: ['png'] },
  { id: 'tiff', label: 'TIFF', extensions: ['tif', 'tiff'] },
];

export const clampExportDimension = (value: number): number =>
  Math.max(1, Math.min(65_535, Math.round(Number.isFinite(value) ? value : 1)));

export const clampExportPercent = (value: number): number =>
  Math.max(1, Math.min(1_600, Math.round(Number.isFinite(value) ? value : 100)));

export const dimensionsFromPercent = (
  sourceWidth: number,
  sourceHeight: number,
  percent: number,
): { height: number; width: number } => {
  const scale = clampExportPercent(percent) / 100;
  return {
    width: clampExportDimension(sourceWidth * scale),
    height: clampExportDimension(sourceHeight * scale),
  };
};

export const heightFromWidth = (sourceWidth: number, sourceHeight: number, width: number): number =>
  clampExportDimension((clampExportDimension(width) * sourceHeight) / Math.max(1, sourceWidth));

export const widthFromHeight = (sourceWidth: number, sourceHeight: number, height: number): number =>
  clampExportDimension((clampExportDimension(height) * sourceWidth) / Math.max(1, sourceHeight));

const normalizeOptionalText = (value: string): string | null => {
  const normalized = value.trim();
  return normalized.length > 0 ? normalized : null;
};

export const buildMetadataOverrides = (settings: ExportDialogSettings): ExportMetadataOverrides | null => {
  if (settings.metadataMode === 'none') return null;

  const exportText = (field: 'artist' | 'contact' | 'copyright' | 'description'): string | null => {
    if (settings.metadataMode === 'all') {
      return settings.metadataEditedFields[field] ? settings[field].trim() : null;
    }
    return normalizeOptionalText(settings[field]);
  };
  const overrides: ExportMetadataOverrides = {
    artist: exportText('artist'),
    contact: exportText('contact'),
    copyright: exportText('copyright'),
    description: settings.metadataMode === 'all' ? exportText('description') : null,
  };
  const hasExplicitAllMetadataEdit =
    settings.metadataMode === 'all' && Object.values(settings.metadataEditedFields).some(Boolean);
  return hasExplicitAllMetadataEdit || Object.values(overrides).some(Boolean) ? overrides : null;
};

export const buildBackendExportSettings = (
  settings: ExportDialogSettings,
  filenameTemplate: string | null,
): ExportSettings => {
  const widthChanged = settings.resizeWidth !== settings.sourceWidth;
  const heightChanged = settings.resizeHeight !== settings.sourceHeight;
  return {
    embedColorProfile: settings.embedColorProfile,
    filenameTemplate,
    jpegQuality: settings.jpegQuality,
    keepMetadata: settings.metadataMode === 'all',
    metadataOverrides: buildMetadataOverrides(settings),
    preserveTimestamps: false,
    resize:
      widthChanged || heightChanged
        ? {
            mode: widthChanged ? 'width' : 'height',
            value: clampExportDimension(widthChanged ? settings.resizeWidth : settings.resizeHeight),
            dontEnlarge: false,
          }
        : null,
    stripGps: settings.stripGps,
    watermark: null,
    exportMasks: false,
    preserveFolders: false,
  };
};

const readMetadataText = (metadata: Record<string, unknown> | null | undefined, aliases: string[]): string => {
  if (!metadata) return '';
  const entries = Object.entries(metadata);
  for (const alias of aliases) {
    const match = entries.find(([key]) => key.toLowerCase() === alias.toLowerCase());
    if (match?.[1] !== null && match?.[1] !== undefined) return String(match[1]);
  }
  return '';
};

export const createInitialExportDialogSettings = (
  source: Pick<ExportDialogSource, 'height' | 'width'>,
  initialFormat: ExportDialogFormat,
  metadata?: Record<string, unknown> | null,
): ExportDialogSettings => ({
  artist: readMetadataText(metadata, ['Artist', 'Creator', 'Author']),
  contact: readMetadataText(metadata, ['Contact', 'OwnerName']),
  copyright: readMetadataText(metadata, ['Copyright']),
  description: readMetadataText(metadata, ['ImageDescription', 'Description', 'Caption']),
  embedColorProfile: true,
  format: initialFormat,
  jpegQuality: 95,
  metadataEditedFields: {
    artist: false,
    contact: false,
    copyright: false,
    description: false,
  },
  metadataMode: 'all',
  resizeHeight: clampExportDimension(source.height),
  resizePercent: 100,
  resizeWidth: clampExportDimension(source.width),
  sourceHeight: clampExportDimension(source.height),
  sourceWidth: clampExportDimension(source.width),
  stripGps: true,
});

export const exportDialogExtension = (format: ExportDialogFormat): string =>
  EXPORT_DIALOG_FORMATS.find((candidate) => candidate.id === format)?.extensions[0] ?? 'jpg';

export const ensureExportPathExtension = (outputPath: string, format: ExportDialogFormat): string => {
  const candidate = EXPORT_DIALOG_FORMATS.find((entry) => entry.id === format);
  const canonicalExtension = candidate?.extensions[0] ?? 'jpg';
  const separatorIndex = Math.max(outputPath.lastIndexOf('/'), outputPath.lastIndexOf('\\'));
  const extensionIndex = outputPath.lastIndexOf('.');
  const currentExtension =
    extensionIndex > separatorIndex + 1 ? outputPath.slice(extensionIndex + 1).toLowerCase() : '';

  if (candidate?.extensions.includes(currentExtension)) return outputPath;
  const stem = extensionIndex > separatorIndex + 1 ? outputPath.slice(0, extensionIndex) : outputPath;
  return `${stem}.${canonicalExtension}`;
};

export const buildSuggestedExportPath = (sourcePath: string, suffix: string, format: ExportDialogFormat): string => {
  const physicalPath = sourcePath.split('?vc=')[0];
  const separatorIndex = Math.max(physicalPath.lastIndexOf('/'), physicalPath.lastIndexOf('\\'));
  const parent = separatorIndex >= 0 ? physicalPath.slice(0, separatorIndex + 1) : '';
  const fileName = separatorIndex >= 0 ? physicalPath.slice(separatorIndex + 1) : physicalPath;
  const extensionIndex = fileName.lastIndexOf('.');
  const stem = extensionIndex > 0 ? fileName.slice(0, extensionIndex) : fileName || 'image';
  return `${parent}${stem}${suffix}.${exportDialogExtension(format)}`;
};
