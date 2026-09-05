import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';

import MainLibrary from '../panel/MainLibrary';
import BottomBar from '../panel/BottomBar';
import LibraryQuickPreview from '../panel/library/LibraryQuickPreview';

import { useUIStore } from '../../store/useUIStore';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useEditorStore } from '../../store/useEditorStore';
import { useProcessStore } from '../../store/useProcessStore';
import { useSettingsStore } from '../../store/useSettingsStore';

import { ImageFile, LibraryViewMode, Panel, ThumbnailAspectRatio, ThumbnailSize } from '../ui/AppProperties';
import { GroupBadgeInfo, GroupId } from '../../utils/imageGrouping';

interface LibraryViewProps {
  sortedImageList: ImageFile[];
  groupBadgeInfo: Map<GroupId, GroupBadgeInfo> | null;
  thumbnailSize: ThumbnailSize;
  thumbnailAspectRatio: ThumbnailAspectRatio;
  libraryViewMode: LibraryViewMode;
  isAndroid: boolean;
  setThumbnailSize: (size: ThumbnailSize) => void;
  setThumbnailAspectRatio: (ratio: ThumbnailAspectRatio) => void;
  setLibraryViewMode: (mode: LibraryViewMode) => void;
  handleClearSelection: () => void;
  handleLibraryImageSingleClick: (...args: any) => void;
  handleImageSelect: (...args: any) => void;
  handleRate: (...args: any) => void;
  handleThumbnailContextMenu: (...args: any) => void;
  handleMainLibraryContextMenu: (...args: any) => void;
  handleGoHome: (...args: any) => void;
  handleOpenImage: (...args: any) => void;
  handleOpenBatchGeometryWorkflow: () => void;
  handleOpenMultiImageWorkflow: () => void;
  handleOpenFolder: (...args: any) => void;
  handleImportClick: (path: string) => void;
  handleLibraryRefresh: () => Promise<void>;
  handleCopyAdjustments: () => void;
  handlePasteAdjustments: () => void;
  handleResetAdjustments: () => void;
  requestThumbnails: any;
}

export default function LibraryView({
  sortedImageList,
  groupBadgeInfo,
  thumbnailSize,
  thumbnailAspectRatio,
  libraryViewMode,
  isAndroid,
  setThumbnailSize,
  setThumbnailAspectRatio,
  setLibraryViewMode,
  handleClearSelection,
  handleLibraryImageSingleClick,
  handleImageSelect,
  handleRate,
  handleThumbnailContextMenu,
  handleMainLibraryContextMenu,
  handleGoHome,
  handleOpenImage,
  handleOpenBatchGeometryWorkflow,
  handleOpenMultiImageWorkflow,
  handleOpenFolder,
  handleImportClick,
  handleLibraryRefresh,
  handleCopyAdjustments,
  handlePasteAdjustments,
  handleResetAdjustments,
  requestThumbnails,
}: LibraryViewProps) {
  const { isLibraryQuickPreviewOpen, libraryContextPanel, setUI } = useUIStore(
    useShallow((state) => ({
      isLibraryQuickPreviewOpen: state.isLibraryQuickPreviewOpen,
      libraryContextPanel: state.libraryContextPanel,
      setUI: state.setUI,
    })),
  );

  const {
    rootPaths,
    currentFolderPath,
    libraryActivePath,
    multiSelectedPaths,
    imageList,
    imageRatings,
    isViewLoading,
    isTreeLoading,
    contentState,
    setLibrary,
  } = useLibraryStore(
    useShallow((state) => ({
      rootPaths: state.rootPaths,
      currentFolderPath: state.currentFolderPath,
      libraryActivePath: state.libraryActivePath,
      multiSelectedPaths: state.multiSelectedPaths,
      imageList: state.imageList,
      imageRatings: state.imageRatings,
      isViewLoading: state.isViewLoading,
      isTreeLoading: state.isTreeLoading,
      contentState: state.contentState,
      setLibrary: state.setLibrary,
    })),
  );

  const { appSettings, handleSettingsChange } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );

  const { aiModelDownloadStatus, importState, indexingProgress, isIndexing, thumbnailProgress, isCopied, isPasted } =
    useProcessStore(
      useShallow((state) => ({
        aiModelDownloadStatus: state.aiModelDownloadStatus,
        importState: state.importState,
        indexingProgress: state.indexingProgress,
        isIndexing: state.isIndexing,
        thumbnailProgress: state.thumbnailProgress,
        isCopied: state.isCopied,
        isPasted: state.isPasted,
      })),
    );

  const activeImageIndex = sortedImageList.findIndex((image) => image.path === libraryActivePath);
  const activeImage = activeImageIndex >= 0 ? sortedImageList[activeImageIndex] : null;

  useEffect(() => {
    if (isLibraryQuickPreviewOpen && !activeImage) {
      setUI({ isLibraryQuickPreviewOpen: false });
    }
  }, [activeImage, isLibraryQuickPreviewOpen, setUI]);

  const handleEnterEdit = () => {
    if (libraryActivePath) void handleImageSelect(libraryActivePath, true);
  };

  const handleToggleInfo = () => {
    if (!libraryActivePath) return;
    if (useEditorStore.getState().selectedImage?.path !== libraryActivePath) {
      void handleImageSelect(libraryActivePath, false);
    }
    setUI({ libraryContextPanel: libraryContextPanel === Panel.Metadata ? null : Panel.Metadata });
  };

  const handlePreviewNavigate = (direction: -1 | 1) => {
    if (sortedImageList.length === 0) return;
    const currentIndex = activeImageIndex >= 0 ? activeImageIndex : 0;
    const nextIndex = (currentIndex + direction + sortedImageList.length) % sortedImageList.length;
    const nextPath = sortedImageList[nextIndex].path;
    setLibrary({
      libraryActivePath: nextPath,
      multiSelectedPaths: [nextPath],
      selectionAnchorPath: nextPath,
    });
  };

  return (
    <div className="relative flex flex-row grow h-full min-h-0">
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        <MainLibrary
          activePath={libraryActivePath}
          aiModelDownloadStatus={aiModelDownloadStatus}
          appSettings={appSettings}
          currentFolderPath={currentFolderPath}
          groupBadgeInfo={groupBadgeInfo}
          imageList={sortedImageList}
          imageRatings={imageRatings}
          importState={importState}
          indexingProgress={indexingProgress}
          isIndexing={isIndexing}
          isLoading={isViewLoading}
          isTreeLoading={isTreeLoading}
          contentState={contentState}
          isAndroid={isAndroid}
          libraryViewMode={libraryViewMode}
          multiSelectedPaths={multiSelectedPaths}
          onClearSelection={handleClearSelection}
          onContextMenu={handleThumbnailContextMenu}
          onEmptyAreaContextMenu={handleMainLibraryContextMenu}
          onGoHome={handleGoHome}
          onImageClick={handleLibraryImageSingleClick}
          onImageDoubleClick={(path) => handleImageSelect(path, true)}
          onImportClick={() => handleImportClick(currentFolderPath as string)}
          onLibraryRefresh={handleLibraryRefresh}
          onRate={handleRate}
          onOpenImage={handleOpenImage}
          onOpenBatchGeometryWorkflow={handleOpenBatchGeometryWorkflow}
          onOpenMultiImageWorkflow={handleOpenMultiImageWorkflow}
          onOpenFolder={handleOpenFolder}
          onSettingsChange={handleSettingsChange}
          onThumbnailAspectRatioChange={setThumbnailAspectRatio}
          onThumbnailSizeChange={setThumbnailSize}
          onRequestThumbnails={requestThumbnails}
          rootPaths={rootPaths}
          setLibraryViewMode={setLibraryViewMode}
          thumbnailAspectRatio={thumbnailAspectRatio}
          thumbnailProgress={thumbnailProgress}
          thumbnailSize={thumbnailSize}
          totalImageCount={imageList.length}
        />
        {((rootPaths && rootPaths.length > 0) || imageList.length > 0) && (
          <BottomBar
            isCopied={isCopied}
            isCopyDisabled={multiSelectedPaths.length !== 1}
            isExportDisabled={multiSelectedPaths.length === 0}
            isLibraryView={true}
            isPasted={isPasted}
            isPasteDisabled={useEditorStore.getState().copiedAdjustments === null || multiSelectedPaths.length === 0}
            isRatingDisabled={multiSelectedPaths.length === 0}
            isResetDisabled={multiSelectedPaths.length === 0}
            multiSelectedPaths={multiSelectedPaths}
            onCopy={handleCopyAdjustments}
            onExportClick={() =>
              setUI({ libraryContextPanel: libraryContextPanel === Panel.Export ? null : Panel.Export })
            }
            onEditSelected={handleEnterEdit}
            onInfoClick={handleToggleInfo}
            onOpenCopyPasteSettings={() => setUI({ isCopyPasteSettingsModalOpen: true })}
            onPaste={() => handlePasteAdjustments()}
            onRate={handleRate}
            onQuickPreview={() => setUI({ isLibraryQuickPreviewOpen: true, libraryContextPanel: null })}
            onReset={() => handleResetAdjustments()}
            rating={imageRatings[libraryActivePath || ''] || 0}
            thumbnailAspectRatio={thumbnailAspectRatio}
            totalImages={imageList.length}
          />
        )}
      </div>
      {isLibraryQuickPreviewOpen && activeImage && (
        <LibraryQuickPreview
          image={activeImage}
          index={activeImageIndex}
          onClose={() => setUI({ isLibraryQuickPreviewOpen: false })}
          onEdit={handleEnterEdit}
          onNavigate={handlePreviewNavigate}
          onRate={handleRate}
          rating={imageRatings[activeImage.path] || 0}
          total={sortedImageList.length}
        />
      )}
    </div>
  );
}
