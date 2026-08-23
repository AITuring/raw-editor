import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { save } from '@tauri-apps/plugin-dialog';
import { useUIStore } from '../store/useUIStore';
import { useSettingsStore } from '../store/useSettingsStore';
import type { ImageStackAlignmentMode, ImageStackBlendMode } from '../store/useUIStore';
import { Invokes } from '../components/ui/AppProperties';
import i18n from '../i18n';
import { IMAGE_STACK_PIPELINE_VERSION } from '../utils/imageStackPipeline';
import { buildSuggestedExportPath, buildBackendExportSettings } from '../features/export/exportDialog';
import type { ExportDialogFormat, ExportDialogSettings } from '../features/export/exportDialog';

const IMAGE_STACK_EXPORT_FORMATS: Record<ExportDialogFormat, { filterName: string; extensions: string[] }> = {
  tiff: { filterName: 'TIFF (16-bit)', extensions: ['tif', 'tiff'] },
  png: { filterName: 'PNG (16-bit)', extensions: ['png'] },
  jpeg: { filterName: 'JPEG', extensions: ['jpg', 'jpeg'] },
};

const getImageStackSuggestedPath = (
  firstPath: string,
  blendMode: ImageStackBlendMode,
  exportFormat: ExportDialogFormat,
) => {
  const suffix = blendMode === 'focus' ? 'FocusStack' : 'Panorama';
  return buildSuggestedExportPath(firstPath, `_${suffix}`, exportFormat);
};

export function useProductivityActions(refreshImageList: () => Promise<void>) {
  const setUI = useUIStore((state) => state.setUI);

  const handleStartPanorama = useCallback(
    (paths: string[]) => {
      setUI((state) => ({
        panoramaModalState: {
          ...state.panoramaModalState,
          isProcessing: true,
          error: null,
          finalImageBase64: null,
          progressMessage: 'Starting panorama process...',
        },
      }));
      invoke(Invokes.StitchPanorama, { paths }).catch((err) => {
        setUI((state) => ({
          panoramaModalState: { ...state.panoramaModalState, isProcessing: false, error: String(err) },
        }));
      });
    },
    [setUI],
  );

  const handleSavePanorama = useCallback(async (): Promise<string> => {
    const { panoramaModalState } = useUIStore.getState();
    if (panoramaModalState.stitchingSourcePaths.length === 0) {
      const err = 'Source paths for panorama not found.';
      setUI((state) => ({ panoramaModalState: { ...state.panoramaModalState, error: err } }));
      throw new Error(err);
    }
    try {
      const savedPath: string = await invoke(Invokes.SavePanorama, {
        firstPathStr: panoramaModalState.stitchingSourcePaths[0],
      });
      await refreshImageList();
      return savedPath;
    } catch (err) {
      console.error('Failed to save panorama:', err);
      setUI((state) => ({ panoramaModalState: { ...state.panoramaModalState, error: String(err) } }));
      throw err;
    }
  }, [refreshImageList, setUI]);

  const handleStartImageStack = useCallback(
    (paths: string[], blendMode: ImageStackBlendMode, alignmentMode: ImageStackAlignmentMode) => {
      const requestId = crypto.randomUUID();
      setUI((state) => ({
        imageStackModalState: {
          ...state.imageStackModalState,
          detailImageBase64: null,
          isProcessing: true,
          error: null,
          finalImageBase64: null,
          progressMessage: 'Starting image alignment process…',
          requestId,
          resultId: null,
          resultSize: null,
          sourcePaths: paths,
          blendMode,
          alignmentMode,
        },
      }));
      invoke(Invokes.ProcessImageStack, {
        paths,
        blendMode,
        alignmentMode,
        pipelineVersion: IMAGE_STACK_PIPELINE_VERSION,
        requestId,
      }).catch((err) => {
        setUI((state) => {
          if (state.imageStackModalState.requestId !== requestId) return state;
          return {
            imageStackModalState: {
              ...state.imageStackModalState,
              isProcessing: false,
              error: String(err),
              resultId: null,
              resultSize: null,
            },
          };
        });
      });
    },
    [setUI],
  );

  const handleSaveImageStack = useCallback(
    async (blendMode: ImageStackBlendMode, settings: ExportDialogSettings): Promise<string | null> => {
      const { imageStackModalState } = useUIStore.getState();
      if (imageStackModalState.sourcePaths.length === 0) {
        const error = 'Source paths for image stack not found.';
        setUI((state) => ({ imageStackModalState: { ...state.imageStackModalState, error } }));
        throw new Error(error);
      }
      if (!imageStackModalState.resultId) {
        const error = 'The visible image-stack preview is not available to save.';
        setUI((state) => ({ imageStackModalState: { ...state.imageStackModalState, error } }));
        throw new Error(error);
      }
      try {
        const firstPath = imageStackModalState.sourcePaths[0];
        const exportFormat = settings.format;
        const format = IMAGE_STACK_EXPORT_FORMATS[exportFormat];
        const outputPath =
          useSettingsStore.getState().osPlatform === 'android'
            ? null
            : await save({
                title: i18n.t('modals.imageStack.save'),
                defaultPath: getImageStackSuggestedPath(firstPath, blendMode, exportFormat),
                filters: [{ name: format.filterName, extensions: format.extensions }],
              });
        if (useSettingsStore.getState().osPlatform !== 'android' && !outputPath) return null;

        const savedPath: string = await invoke(Invokes.SaveImageStack, {
          firstPathStr: firstPath,
          blendMode,
          outputFormat: exportFormat,
          exportSettings: buildBackendExportSettings(settings, null),
          resultId: imageStackModalState.resultId,
          outputPathStr: outputPath,
        });
        try {
          await refreshImageList();
        } catch (refreshError) {
          console.warn('Image stack was saved, but the current library could not be refreshed:', refreshError);
        }
        return savedPath;
      } catch (err) {
        setUI((state) => ({
          imageStackModalState: { ...state.imageStackModalState, error: String(err) },
        }));
        throw err;
      }
    },
    [refreshImageList, setUI],
  );

  const handleStartHdr = useCallback(
    (paths: string[]) => {
      setUI((state) => ({
        hdrModalState: {
          ...state.hdrModalState,
          isProcessing: true,
          error: null,
          finalImageBase64: null,
          progressMessage: 'Starting HDR process...',
        },
      }));
      invoke(Invokes.MergeHdr, { paths }).catch((err) => {
        setUI((state) => ({ hdrModalState: { ...state.hdrModalState, isProcessing: false, error: String(err) } }));
      });
    },
    [setUI],
  );

  const handleSaveHdr = useCallback(async (): Promise<string> => {
    const { hdrModalState } = useUIStore.getState();
    if (hdrModalState.stitchingSourcePaths.length === 0) {
      const err = 'Source paths for HDR not found.';
      setUI((state) => ({ hdrModalState: { ...state.hdrModalState, error: err } }));
      throw new Error(err);
    }
    try {
      const savedPath: string = await invoke(Invokes.SaveHdr, { firstPathStr: hdrModalState.stitchingSourcePaths[0] });
      await refreshImageList();
      return savedPath;
    } catch (err) {
      console.error('Failed to save HDR image:', err);
      setUI((state) => ({ hdrModalState: { ...state.hdrModalState, error: String(err) } }));
      throw err;
    }
  }, [refreshImageList, setUI]);

  const handleApplyDenoise = useCallback(
    async (intensity: number, method: 'ai' | 'bm3d') => {
      const { denoiseModalState } = useUIStore.getState();
      if (denoiseModalState.targetPaths.length === 0) return;

      setUI((state) => ({
        denoiseModalState: {
          ...state.denoiseModalState,
          isProcessing: true,
          error: null,
          progressMessage: 'Starting engine...',
        },
      }));

      try {
        await invoke(Invokes.ApplyDenoising, {
          path: denoiseModalState.targetPaths[0],
          intensity: intensity,
          method: method,
        });
      } catch (err) {
        setUI((state) => ({
          denoiseModalState: { ...state.denoiseModalState, isProcessing: false, error: String(err) },
        }));
      }
    },
    [setUI],
  );

  const handleBatchDenoise = useCallback(
    async (intensity: number, method: 'ai' | 'bm3d', paths: string[]) => {
      try {
        const savedPaths: string[] = await invoke('batch_denoise_images', { paths, intensity, method });
        await refreshImageList();
        return savedPaths;
      } catch (err) {
        setUI((state) => ({ denoiseModalState: { ...state.denoiseModalState, error: String(err) } }));
        throw err;
      }
    },
    [refreshImageList, setUI],
  );

  const handleSaveDenoisedImage = useCallback(async (): Promise<string> => {
    const { denoiseModalState } = useUIStore.getState();
    if (denoiseModalState.targetPaths.length === 0) throw new Error('No target path');
    const savedPath = await invoke<string>(Invokes.SaveDenoisedImage, {
      originalPathStr: denoiseModalState.targetPaths[0],
    });
    await refreshImageList();
    return savedPath;
  }, [refreshImageList]);

  const handleSaveCollage = useCallback(
    async (base64Data: string, firstPath: string): Promise<string> => {
      try {
        const savedPath: string = await invoke(Invokes.SaveCollage, { base64Data, firstPathStr: firstPath });
        await refreshImageList();
        return savedPath;
      } catch (err) {
        console.error('Failed to save collage:', err);
        throw err;
      }
    },
    [refreshImageList],
  );

  return {
    handleStartPanorama,
    handleSavePanorama,
    handleStartImageStack,
    handleSaveImageStack,
    handleStartHdr,
    handleSaveHdr,
    handleApplyDenoise,
    handleBatchDenoise,
    handleSaveDenoisedImage,
    handleSaveCollage,
  };
}
