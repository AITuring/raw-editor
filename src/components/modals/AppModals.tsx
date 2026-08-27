import { useCallback, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { save } from '@tauri-apps/plugin-dialog';
import { useShallow } from 'zustand/react/shallow';
import { useTranslation } from 'react-i18next';
import { useUIStore } from '../../store/useUIStore';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useSettingsStore } from '../../store/useSettingsStore';
import { useProcessStore } from '../../store/useProcessStore';
import { useEditorStore } from '../../store/useEditorStore';
import CopyPasteSettingsModal from './CopyPasteSettingsModal';
import PanoramaModal from './PanoramaModal';
import HdrModal from './HdrModal';
import NegativeConversionModal from './NegativeConversionModal';
import DenoiseModal from './DenoiseModal';
import CreateFolderModal from './CreateFolderModal';
import RenameFolderModal from './RenameFolderModal';
import RenameFileModal from './RenameFileModal';
import ConfirmModal from './ConfirmModal';
import ImportSettingsModal from './ImportSettingsModal';
import CullingModal from './CullingModal';
import CollageModal from './CollageModal';
import ImageStackModal from './ImageStackModal';
import ExportImageDialog from '../../features/export/ExportImageDialog';
import { AppSettings, AlbumItem, AlbumGroup, Invokes } from '../ui/AppProperties';
import { Status } from '../ui/ExportImportProperties';
import { CopyPasteSettings } from '../../utils/adjustments';
import type { ImageStackAlignmentMode, ImageStackBlendMode } from '../../store/useUIStore';
import {
  EXPORT_DIALOG_FORMATS,
  buildBackendExportSettings,
  buildSuggestedExportPath,
  ensureExportPathExtension,
  exportDialogExtension,
} from '../../features/export/exportDialog';
import type { ExportDialogSettings, ExportDialogSource } from '../../features/export/exportDialog';

export interface AppModalsProps {
  requestThumbnails: (paths: string[]) => void;
  handleImageSelect: (path: string) => void;
  handleSavePanorama: () => Promise<string>;
  handleStartPanorama: (paths: string[]) => void;
  handleSaveImageStack: (blendMode: ImageStackBlendMode, settings: ExportDialogSettings) => Promise<string | null>;
  handleStartImageStack: (
    paths: string[],
    blendMode: ImageStackBlendMode,
    alignmentMode: ImageStackAlignmentMode,
  ) => void;
  handleSaveHdr: () => Promise<string>;
  handleStartHdr: (paths: string[]) => void;
  refreshImageList: () => Promise<void>;
  handleApplyDenoise: (intensity: number, method: 'ai' | 'bm3d') => Promise<void>;
  handleBatchDenoise: (intensity: number, method: 'ai' | 'bm3d', paths: string[]) => Promise<string[]>;
  handleSaveDenoisedImage: () => Promise<string>;
  handleCreateFolder: (folderName: string) => Promise<void>;
  handleRenameFolder: (newName: string) => Promise<void>;
  handleSaveRename: (nameTemplate: string) => Promise<void>;
  handleStartImport: (settings: any) => Promise<void>;
  handleSetColorLabel: (color: string | null, paths?: string[]) => Promise<void>;
  handleRate: (rating: number, paths?: string[]) => void;
  executeDelete: (paths: string[], options: any) => Promise<void>;
  handleSaveCollage: (base64Data: string, firstPath: string) => Promise<string>;
  handleCreateAlbumItem: (name: string, type: 'album' | 'group') => Promise<void>;
  handleRenameAlbumItem: (newName: string) => Promise<void>;
}

export default function AppModals(props: AppModalsProps) {
  const { t } = useTranslation();
  const { appSettings, handleSettingsChange, osPlatform } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      handleSettingsChange: state.handleSettingsChange,
      osPlatform: state.osPlatform,
    })),
  );

  const {
    isCreateFolderModalOpen,
    isRenameFolderModalOpen,
    isRenameFileModalOpen,
    isImportModalOpen,
    isCopyPasteSettingsModalOpen,
    folderActionTarget,
    renameTargetPaths,
    importSourcePaths,
    isCreateAlbumModalOpen,
    isCreateAlbumGroupModalOpen,
    isRenameAlbumModalOpen,
    albumActionTarget,
    confirmModalState,
    panoramaModalState,
    imageStackModalState,
    hdrModalState,
    negativeModalState,
    denoiseModalState,
    cullingModalState,
    collageModalState,
    isEditorExportDialogOpen,
    setUI,
  } = useUIStore(
    useShallow((state) => ({
      isCreateFolderModalOpen: state.isCreateFolderModalOpen,
      isRenameFolderModalOpen: state.isRenameFolderModalOpen,
      isRenameFileModalOpen: state.isRenameFileModalOpen,
      isImportModalOpen: state.isImportModalOpen,
      isCopyPasteSettingsModalOpen: state.isCopyPasteSettingsModalOpen,
      folderActionTarget: state.folderActionTarget,
      renameTargetPaths: state.renameTargetPaths,
      importSourcePaths: state.importSourcePaths,
      isCreateAlbumModalOpen: state.isCreateAlbumModalOpen,
      isCreateAlbumGroupModalOpen: state.isCreateAlbumGroupModalOpen,
      isRenameAlbumModalOpen: state.isRenameAlbumModalOpen,
      albumActionTarget: state.albumActionTarget,
      confirmModalState: state.confirmModalState,
      panoramaModalState: state.panoramaModalState,
      imageStackModalState: state.imageStackModalState,
      hdrModalState: state.hdrModalState,
      negativeModalState: state.negativeModalState,
      denoiseModalState: state.denoiseModalState,
      cullingModalState: state.cullingModalState,
      collageModalState: state.collageModalState,
      isEditorExportDialogOpen: state.isEditorExportDialogOpen,
      setUI: state.setUI,
    })),
  );

  const { thumbnails, aiModelDownloadStatus, setExportState } = useProcessStore(
    useShallow((state) => ({
      thumbnails: state.thumbnails,
      aiModelDownloadStatus: state.aiModelDownloadStatus,
      setExportState: state.setExportState,
    })),
  );

  const { selectedImage, finalPreviewUrl, adjustments } = useEditorStore(
    useShallow((state) => ({
      selectedImage: state.selectedImage,
      finalPreviewUrl: state.finalPreviewUrl,
      adjustments: state.adjustments,
    })),
  );

  const { imageList, rootPaths } = useLibraryStore(
    useShallow((state) => ({
      imageList: state.imageList,
      rootPaths: state.rootPaths,
    })),
  );

  const editorExportSource = useMemo<ExportDialogSource | null>(() => {
    if (!selectedImage) return null;
    const previewSrc = finalPreviewUrl || selectedImage.originalUrl || selectedImage.thumbnailUrl;
    if (!previewSrc) return null;

    const crop = adjustments.crop;
    const orientationSteps = adjustments.orientationSteps || 0;
    const isSwapped = orientationSteps === 1 || orientationSteps === 3;
    const sourceWidth = crop?.width ?? (isSwapped ? selectedImage.height : selectedImage.width);
    const sourceHeight = crop?.height ?? (isSwapped ? selectedImage.width : selectedImage.height);

    return {
      detailPreviewSrc: finalPreviewUrl || selectedImage.originalUrl || previewSrc,
      fileName: selectedImage.path.split('?')[0].split(/[\\/]/).pop() || selectedImage.path,
      height: Math.max(1, Math.round(sourceHeight)),
      previewSrc,
      width: Math.max(1, Math.round(sourceWidth)),
    };
  }, [adjustments.crop, adjustments.orientationSteps, finalPreviewUrl, selectedImage]);

  const imageStackSourceMetadata = useMemo<Record<string, unknown> | null>(() => {
    const firstPath = imageStackModalState.sourcePaths[0];
    if (!firstPath) return null;
    const physicalPath = firstPath.split('?')[0];
    return (
      imageList.find((image) => image.path === firstPath || image.path.split('?')[0] === physicalPath)?.exif ?? null
    );
  }, [imageList, imageStackModalState.sourcePaths]);

  const handleEditorExport = useCallback(
    async (settings: ExportDialogSettings): Promise<string | null> => {
      const currentImage = useEditorStore.getState().selectedImage;
      if (!currentImage) throw new Error(t('export.status.noImageSelected'));

      const format = EXPORT_DIALOG_FORMATS.find((candidate) => candidate.id === settings.format);
      if (!format) throw new Error(`Unsupported export format: ${settings.format}`);

      const suggestedPath = buildSuggestedExportPath(currentImage.path, '_edited', settings.format);
      const selectedOutputPath =
        osPlatform === 'android'
          ? suggestedPath.split(/[\\/]/).pop() || suggestedPath
          : await save({
              defaultPath: suggestedPath,
              filters: [{ name: format.label, extensions: format.extensions }],
              title: t('export.dialog.saveEditedImageTitle'),
            });
      if (!selectedOutputPath) return null;
      const outputPath = ensureExportPathExtension(selectedOutputPath, settings.format);

      setExportState({
        errorMessage: '',
        progress: { current: 0, total: 1 },
        status: Status.Exporting,
      });

      try {
        await invoke(Invokes.ExportImages, {
          baseOriginFolders: rootPaths,
          currentEditAdjustments: useEditorStore.getState().adjustments,
          currentEditPath: currentImage.path,
          exportSettings: buildBackendExportSettings(settings, '{original_filename}_edited'),
          isExplicitFilePath: true,
          outputFolderOrFile: outputPath,
          outputFormat: exportDialogExtension(settings.format),
          paths: [currentImage.path],
          waitForCompletion: true,
        });
        return outputPath;
      } catch (exportError) {
        const errorMessage = exportError instanceof Error ? exportError.message : String(exportError);
        setExportState({ errorMessage, status: Status.Error });
        throw exportError;
      }
    },
    [osPlatform, rootPaths, setExportState, t],
  );

  const handleEstimateEditorExportSize = useCallback(async (settings: ExportDialogSettings): Promise<number | null> => {
    const currentEditor = useEditorStore.getState();
    if (!currentEditor.selectedImage) return null;

    return invoke<number>(Invokes.EstimateExportSizes, {
      currentEditAdjustments: currentEditor.adjustments,
      currentEditPath: currentEditor.selectedImage.path,
      exportSettings: buildBackendExportSettings(settings, '{original_filename}_edited'),
      outputFormat: settings.format,
      paths: [currentEditor.selectedImage.path],
    });
  }, []);

  const closeConfirmModal = () => {
    setUI((state) => ({ confirmModalState: { ...state.confirmModalState, isOpen: false } }));
  };

  const closeImageStackModal = () => {
    setUI({
      imageStackModalState: {
        detailImageBase64: null,
        isOpen: false,
        isProcessing: false,
        progressMessage: null,
        finalImageBase64: null,
        error: null,
        requestId: null,
        resultId: null,
        resultSize: null,
        sourcePaths: [],
        blendMode: 'focus',
        alignmentMode: 'auto',
      },
    });
  };

  const currentAlbumData = (() => {
    if (!albumActionTarget) return null;
    const { albumTree } = useLibraryStore.getState();
    const findNode = (nodes: AlbumItem[]): AlbumItem | null => {
      for (const n of nodes) {
        if (n.id === albumActionTarget) return n;
        if (n.type === 'group') {
          const res = findNode((n as AlbumGroup).children);
          if (res) return res;
        }
      }
      return null;
    };
    return findNode(albumTree);
  })();

  const currentAlbumName = currentAlbumData?.name || '';
  const isAlbumGroup = currentAlbumData?.type === 'group';

  return (
    <>
      <ExportImageDialog
        initialFormat="jpeg"
        isOpen={isEditorExportDialogOpen}
        metadata={selectedImage?.exif ?? null}
        onClose={() => setUI({ isEditorExportDialogOpen: false })}
        onEstimateSize={handleEstimateEditorExportSize}
        onExport={handleEditorExport}
        source={editorExportSource}
      />
      <CopyPasteSettingsModal
        isOpen={isCopyPasteSettingsModalOpen}
        onClose={() => setUI({ isCopyPasteSettingsModalOpen: false })}
        settings={appSettings?.copyPasteSettings as CopyPasteSettings}
        onSave={(newSettings) =>
          handleSettingsChange({ ...appSettings, copyPasteSettings: newSettings } as AppSettings)
        }
      />
      <PanoramaModal
        error={panoramaModalState.error}
        finalImageBase64={panoramaModalState.finalImageBase64}
        imageCount={panoramaModalState.stitchingSourcePaths.length}
        isOpen={panoramaModalState.isOpen}
        isProcessing={panoramaModalState.isProcessing}
        loadingImageUrl={
          panoramaModalState.stitchingSourcePaths.length > 0
            ? thumbnails[
                panoramaModalState.stitchingSourcePaths[Math.floor(panoramaModalState.stitchingSourcePaths.length / 2)]
              ] || null
            : null
        }
        onClose={() =>
          setUI({
            panoramaModalState: {
              isOpen: false,
              isProcessing: false,
              progressMessage: '',
              finalImageBase64: null,
              error: null,
              stitchingSourcePaths: [],
            },
          })
        }
        onOpenFile={(path: string) => props.handleImageSelect(path)}
        onSave={props.handleSavePanorama}
        onStitch={() => props.handleStartPanorama(panoramaModalState.stitchingSourcePaths)}
        progressMessage={panoramaModalState.progressMessage}
      />
      <ImageStackModal
        detailImageBase64={imageStackModalState.detailImageBase64}
        error={imageStackModalState.error}
        finalImageBase64={imageStackModalState.finalImageBase64}
        initialAlignmentMode={imageStackModalState.alignmentMode}
        initialBlendMode={imageStackModalState.blendMode}
        isOpen={imageStackModalState.isOpen}
        isProcessing={imageStackModalState.isProcessing}
        resultSize={imageStackModalState.resultSize}
        onClose={closeImageStackModal}
        onChange={() =>
          setUI((state) => ({
            imageStackModalState: {
              ...state.imageStackModalState,
              detailImageBase64: null,
              error: null,
              finalImageBase64: null,
              progressMessage: null,
              requestId: null,
              resultId: null,
              resultSize: null,
            },
          }))
        }
        onOpenFile={(path: string) => {
          closeImageStackModal();
          props.handleImageSelect(path);
        }}
        onProcess={props.handleStartImageStack}
        onRequestThumbnails={props.requestThumbnails}
        onSave={props.handleSaveImageStack}
        progressMessage={imageStackModalState.progressMessage}
        sourceMetadata={imageStackSourceMetadata}
        sourcePaths={imageStackModalState.sourcePaths}
        thumbnails={thumbnails}
      />
      <HdrModal
        error={hdrModalState.error}
        finalImageBase64={hdrModalState.finalImageBase64}
        imageCount={hdrModalState.stitchingSourcePaths.length}
        isOpen={hdrModalState.isOpen}
        isProcessing={hdrModalState.isProcessing}
        loadingImageUrl={
          hdrModalState.stitchingSourcePaths.length > 0
            ? thumbnails[
                hdrModalState.stitchingSourcePaths[Math.floor(hdrModalState.stitchingSourcePaths.length / 2)]
              ] || null
            : null
        }
        onClose={() =>
          setUI({
            hdrModalState: {
              isOpen: false,
              isProcessing: false,
              progressMessage: '',
              finalImageBase64: null,
              error: null,
              stitchingSourcePaths: [],
            },
          })
        }
        onOpenFile={(path: string) => props.handleImageSelect(path)}
        onSave={props.handleSaveHdr}
        onMerge={() => props.handleStartHdr(hdrModalState.stitchingSourcePaths)}
        progressMessage={hdrModalState.progressMessage}
      />
      <NegativeConversionModal
        isOpen={negativeModalState.isOpen}
        onClose={() => setUI((state) => ({ negativeModalState: { ...state.negativeModalState, isOpen: false } }))}
        targetPaths={negativeModalState.targetPaths}
        onSave={(savedPaths) => {
          props.refreshImageList().then(() => {
            if (selectedImage && negativeModalState.targetPaths.includes(selectedImage.path) && savedPaths.length > 0) {
              props.handleImageSelect(savedPaths[0]);
            }
          });
        }}
      />
      <DenoiseModal
        isOpen={denoiseModalState.isOpen}
        onClose={() => setUI((state) => ({ denoiseModalState: { ...state.denoiseModalState, isOpen: false } }))}
        onDenoise={props.handleApplyDenoise}
        onBatchDenoise={props.handleBatchDenoise}
        onSave={props.handleSaveDenoisedImage}
        onOpenFile={props.handleImageSelect}
        previewBase64={denoiseModalState.previewBase64}
        originalBase64={denoiseModalState.originalBase64 || null}
        isProcessing={denoiseModalState.isProcessing}
        error={denoiseModalState.error}
        progressMessage={denoiseModalState.progressMessage}
        aiModelDownloadStatus={aiModelDownloadStatus}
        isRaw={denoiseModalState.isRaw}
        targetPaths={denoiseModalState.targetPaths}
        loadingImageUrl={
          denoiseModalState.targetPaths.length > 0
            ? thumbnails[denoiseModalState.targetPaths[0]] ||
              (selectedImage?.path === denoiseModalState.targetPaths[0] ? finalPreviewUrl : null)
            : null
        }
      />
      <CreateFolderModal
        isOpen={isCreateFolderModalOpen}
        onClose={() => setUI({ isCreateFolderModalOpen: false })}
        onSave={props.handleCreateFolder}
      />
      <RenameFolderModal
        currentName={folderActionTarget ? folderActionTarget.split(/[\\/]/).pop() || '' : ''}
        isOpen={isRenameFolderModalOpen}
        onClose={() => setUI({ isRenameFolderModalOpen: false })}
        onSave={props.handleRenameFolder}
      />
      <CreateFolderModal
        isOpen={isCreateAlbumModalOpen}
        onClose={() => setUI({ isCreateAlbumModalOpen: false })}
        onSave={(name) => props.handleCreateAlbumItem(name, 'album')}
        title={t('contextMenus.albums.newAlbum')}
        placeholder={t('modals.createAlbum.placeholder')}
        buttonText={t('modals.createFolder.create')}
      />
      <CreateFolderModal
        isOpen={isCreateAlbumGroupModalOpen}
        onClose={() => setUI({ isCreateAlbumGroupModalOpen: false })}
        onSave={(name) => props.handleCreateAlbumItem(name, 'group')}
        title={t('contextMenus.albums.newGroup')}
        placeholder={t('modals.createGroup.placeholder')}
        buttonText={t('modals.createFolder.create')}
      />
      <RenameFolderModal
        currentName={currentAlbumName}
        isOpen={isRenameAlbumModalOpen}
        onClose={() => setUI({ isRenameAlbumModalOpen: false })}
        onSave={props.handleRenameAlbumItem}
        title={isAlbumGroup ? t('contextMenus.albums.renameGroup') : t('contextMenus.albums.renameAlbum')}
        placeholder={isAlbumGroup ? t('modals.renameGroup.placeholder') : t('modals.renameAlbum.placeholder')}
      />
      <RenameFileModal
        filesToRename={renameTargetPaths}
        isOpen={isRenameFileModalOpen}
        onClose={() => setUI({ isRenameFileModalOpen: false })}
        onSave={props.handleSaveRename}
      />
      <ConfirmModal {...confirmModalState} onClose={closeConfirmModal} />
      <ImportSettingsModal
        fileCount={importSourcePaths.length}
        isOpen={isImportModalOpen}
        onClose={() => setUI({ isImportModalOpen: false })}
        onSave={props.handleStartImport}
      />
      <CullingModal
        isOpen={cullingModalState.isOpen}
        onClose={() =>
          setUI({
            cullingModalState: { isOpen: false, progress: null, suggestions: null, error: null, pathsToCull: [] },
          })
        }
        progress={cullingModalState.progress}
        suggestions={cullingModalState.suggestions}
        error={cullingModalState.error}
        imagePaths={cullingModalState.pathsToCull}
        thumbnails={thumbnails}
        onApply={(action, paths) => {
          if (action === 'reject') {
            props.handleSetColorLabel('red', paths);
          } else if (action === 'rate_zero') {
            props.handleRate(1, paths);
          } else if (action === 'delete') {
            props.executeDelete(paths, { includeAssociated: false });
          }
          setUI({
            cullingModalState: { isOpen: false, progress: null, suggestions: null, error: null, pathsToCull: [] },
          });
        }}
        onError={(err) => {
          setUI((state) => ({ cullingModalState: { ...state.cullingModalState, error: err, progress: null } }));
        }}
      />
      <CollageModal
        isOpen={collageModalState.isOpen}
        onClose={() => setUI({ collageModalState: { isOpen: false, sourceImages: [] } })}
        onSave={props.handleSaveCollage}
        sourceImages={collageModalState.sourceImages}
        thumbnails={thumbnails}
      />
    </>
  );
}
