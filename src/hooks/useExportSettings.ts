import { useState, useMemo, useCallback } from 'react';
import { ExportPreset, WatermarkAnchor } from '../components/ui/ExportImportProperties';
import { DEFAULT_WATERMARK_PATH, normalizeWatermarkPath } from '../features/export/watermark';

const WATERMARK_ANCHORS = new Set<string>(Object.values(WatermarkAnchor));

export function useExportSettings() {
  const [fileFormat, setFileFormat] = useState('jpeg');
  const [jpegQuality, setJpegQuality] = useState(90);
  const [enableResize, setEnableResize] = useState(false);
  const [resizeMode, setResizeMode] = useState('longEdge');
  const [resizeValue, setResizeValue] = useState(2048);
  const [dontEnlarge, setDontEnlarge] = useState(true);
  const [keepMetadata, setKeepMetadata] = useState(true);
  const [preserveTimestamps, setPreserveTimestamps] = useState(false);
  const [stripGps, setStripGps] = useState(true);
  const [exportMasks, setExportMasks] = useState(false);
  const [preserveFolders, setPreserveFolders] = useState(false);
  const [filenameTemplate, setFilenameTemplate] = useState('{original_filename}_edited');
  const [enableWatermark, setEnableWatermark] = useState(false);
  const [watermarkPath, setWatermarkPath] = useState(DEFAULT_WATERMARK_PATH);
  const [watermarkAnchor, setWatermarkAnchor] = useState<WatermarkAnchor>(WatermarkAnchor.Center);
  const [watermarkScale, setWatermarkScale] = useState(10);
  const [watermarkSpacing, setWatermarkSpacing] = useState(5);
  const [watermarkOpacity, setWatermarkOpacity] = useState(80);

  const handleApplyPreset = useCallback((preset: ExportPreset) => {
    const usesLegacyEmptyWatermark = !preset.watermarkPath;
    setFileFormat(preset.fileFormat);
    setJpegQuality(preset.jpegQuality);
    setEnableResize(preset.enableResize);
    setResizeMode(preset.resizeMode);
    setResizeValue(preset.resizeValue);
    setDontEnlarge(preset.dontEnlarge);
    setKeepMetadata(preset.keepMetadata);
    setPreserveTimestamps(preset.preserveTimestamps ?? false);
    setStripGps(preset.stripGps);
    setExportMasks(preset.exportMasks ?? false);
    setPreserveFolders(preset.preserveFolders ?? false);
    setFilenameTemplate(preset.filenameTemplate);
    setEnableWatermark(preset.enableWatermark);
    setWatermarkPath(normalizeWatermarkPath(preset.watermarkPath));
    setWatermarkAnchor(
      !usesLegacyEmptyWatermark && WATERMARK_ANCHORS.has(preset.watermarkAnchor)
        ? (preset.watermarkAnchor as WatermarkAnchor)
        : WatermarkAnchor.Center,
    );
    setWatermarkScale(preset.watermarkScale ?? 10);
    setWatermarkSpacing(preset.watermarkSpacing ?? 5);
    setWatermarkOpacity(usesLegacyEmptyWatermark ? 80 : (preset.watermarkOpacity ?? 80));
  }, []);

  const currentSettingsObject = useMemo(
    () => ({
      fileFormat,
      jpegQuality,
      enableResize,
      resizeMode,
      resizeValue,
      dontEnlarge,
      keepMetadata,
      preserveTimestamps,
      stripGps,
      exportMasks,
      preserveFolders,
      filenameTemplate,
      enableWatermark,
      watermarkPath,
      watermarkAnchor,
      watermarkScale,
      watermarkSpacing,
      watermarkOpacity,
    }),
    [
      fileFormat,
      jpegQuality,
      enableResize,
      resizeMode,
      resizeValue,
      dontEnlarge,
      keepMetadata,
      preserveTimestamps,
      stripGps,
      exportMasks,
      preserveFolders,
      filenameTemplate,
      enableWatermark,
      watermarkPath,
      watermarkAnchor,
      watermarkScale,
      watermarkSpacing,
      watermarkOpacity,
    ],
  );

  return {
    fileFormat,
    setFileFormat,
    jpegQuality,
    setJpegQuality,
    enableResize,
    setEnableResize,
    resizeMode,
    setResizeMode,
    resizeValue,
    setResizeValue,
    dontEnlarge,
    setDontEnlarge,
    keepMetadata,
    setKeepMetadata,
    preserveTimestamps,
    setPreserveTimestamps,
    stripGps,
    setStripGps,
    exportMasks,
    setExportMasks,
    preserveFolders,
    setPreserveFolders,
    filenameTemplate,
    setFilenameTemplate,
    enableWatermark,
    setEnableWatermark,
    watermarkPath,
    setWatermarkPath,
    watermarkAnchor,
    setWatermarkAnchor,
    watermarkScale,
    setWatermarkScale,
    watermarkSpacing,
    setWatermarkSpacing,
    watermarkOpacity,
    setWatermarkOpacity,
    handleApplyPreset,
    currentSettingsObject,
  };
}
