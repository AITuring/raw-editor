import { type PointerEvent as ReactPointerEvent, useState, useEffect, useCallback, useRef } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { listen, TauriEvent } from '@tauri-apps/api/event';
import { DndContext, DragOverlay, PointerSensor, useSensor, useSensors } from '@dnd-kit/core';
import clsx from 'clsx';
import { Images } from 'lucide-react';

import TitleBar from './window/TitleBar';
import FolderTree from './components/panel/right/FolderTree';
import SettingsPanel from './components/panel/SettingsPanel';
import ExportPanel from './components/panel/right/ExportPanel';
import GlobalTooltip from './components/ui/GlobalTooltip';
import { MessageHost } from './components/ui/Message';
import { message } from './components/ui/messageApi';
import AppModals from './components/modals/AppModals';

import SidePanelArea from './components/panel/SidePanelArea';
import DevelopPanel from './components/panel/DevelopPanel';
import { PANEL_ICONS } from './components/panel/PanelSwitcher';
import Controls from './components/panel/right/ControlsPanel';
import MetadataPanel from './components/panel/right/MetadataPanel';
import CropPanel from './components/panel/right/CropPanel';
import MasksPanel from './components/panel/right/MasksPanel';
import AIPanel from './components/panel/right/AIPanel';
import PresetsPanel from './components/panel/right/PresetsPanel';

import EditorView from './components/views/EditorView';
import LibraryView from './components/views/LibraryView';

import { ContextMenuProvider } from './context/ContextMenuContext';
import { useSettingsStore } from './store/useSettingsStore';
import { useUIStore } from './store/useUIStore';
import { useLibraryStore } from './store/useLibraryStore';
import { useEditorStore } from './store/useEditorStore';
import { useProcessStore } from './store/useProcessStore';
import { useShallow } from 'zustand/react/shallow';

import { useThumbnails } from './hooks/useThumbnails';
import { ImageDimensions } from './hooks/useImageRenderSize';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useTauriListeners } from './hooks/useTauriListeners';
import { useFileOperations } from './hooks/useFileOperations';
import { useAppContextMenus } from './hooks/useAppContextMenus';
import { useSortedLibrary } from './hooks/useSortedLibrary';
import { useAppNavigation } from './hooks/useAppNavigation';
import { useExternalEditSession } from './hooks/useExternalEditSession';
import ExternalEditBar from './components/ui/ExternalEditBar';
import { Status } from './components/ui/ExportImportProperties';

import { useEditorActions } from './hooks/useEditorActions';
import { useLibraryActions } from './hooks/useLibraryActions';
import { useProductivityActions } from './hooks/useProductivityActions';

import { useAppInitialization } from './hooks/useAppInitialization';
import { useAndroidBackHandler } from './hooks/useAndroidBackHandler';
import i18n from './i18n';

import {
  Invokes,
  ImageFile,
  LibraryViewMode,
  Panel,
  PanelRegion,
  ThumbnailSize,
  ThumbnailAspectRatio,
} from './components/ui/AppProperties';

import ImageProcessingManager from './components/managers/ImageProcessingManager';
import ImageLoaderManager from './components/managers/ImageLoaderManager';
import { BASIC_MODE } from './basic/runtime';

const insertChildrenIntoTree = (node: any, targetPath: string, newChildren: any[]): any => {
  if (!node) return null;

  if (node.path === targetPath) {
    const mergedChildren = newChildren.map((newChild: any) => {
      const existingChild = node.children?.find((c: any) => c.path === newChild.path);
      if (existingChild && existingChild.children && existingChild.children.length > 0) {
        return { ...newChild, children: existingChild.children };
      }
      return newChild;
    });
    return { ...node, children: mergedChildren };
  }

  if (node.children && node.children.length > 0) {
    return {
      ...node,
      children: node.children.map((child: any) => insertChildrenIntoTree(child, targetPath, newChildren)),
    };
  }

  return node;
};

function App() {
  const COMPACT_EDITOR_MAX_WIDTH = 900;

  const { appSettings, osPlatform, handleSettingsChange } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      osPlatform: state.osPlatform,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );

  const {
    activeView,
    isFullScreen,
    isWindowFullScreen,
    isInstantTransition,
    isLayoutReady,
    leftPanelWidth,
    rightPanelWidth,
    compactEditorPanelHeightOverride,
    activeRightPanel,
    activeLayoutDragItem,
    isSettingsOpen,
    setUI,
    setRightPanel,
    setLayoutDragItem,
    movePanel,
  } = useUIStore(
    useShallow((state) => ({
      activeView: state.activeView,
      isFullScreen: state.isFullScreen,
      isWindowFullScreen: state.isWindowFullScreen,
      isInstantTransition: state.isInstantTransition,
      isLayoutReady: state.isLayoutReady,
      uiVisibility: state.uiVisibility,
      isLibraryExportPanelVisible: state.isLibraryExportPanelVisible,
      leftPanelWidth: state.leftPanelWidth,
      rightPanelWidth: state.rightPanelWidth,
      compactEditorPanelHeightOverride: state.compactEditorPanelHeightOverride,
      activeRightPanel: state.activeRightPanel,
      activeLayoutDragItem: state.activeLayoutDragItem,
      isSettingsOpen: state.isSettingsOpen,
      setUI: state.setUI,
      setRightPanel: state.setRightPanel,
      setLayoutDragItem: state.setLayoutDragItem,
      movePanel: state.movePanel,
    })),
  );

  const { rootPaths, currentFolderPath, expandedFolders, multiSelectedPaths, libraryImageCount, setLibrary } =
    useLibraryStore(
      useShallow((state) => ({
        rootPaths: state.rootPaths,
        currentFolderPath: state.currentFolderPath,
        expandedFolders: state.expandedFolders,
        multiSelectedPaths: state.multiSelectedPaths,
        libraryImageCount: state.imageList.length,
        setLibrary: state.setLibrary,
      })),
    );

  const { selectedImage, activeMaskContainerId, activeAiPatchContainerId, hasRenderedFirstFrame, setEditor } =
    useEditorStore(
      useShallow((state) => ({
        selectedImage: state.selectedImage,
        activeMaskContainerId: state.activeMaskContainerId,
        activeAiPatchContainerId: state.activeAiPatchContainerId,
        hasRenderedFirstFrame: state.hasRenderedFirstFrame,
        setEditor: state.setEditor,
      })),
    );

  const { exportState, setExportState } = useProcessStore(
    useShallow((state) => ({
      exportState: state.exportState,
      setExportState: state.setExportState,
    })),
  );

  const defaultThumbnailSize = osPlatform === 'android' ? ThumbnailSize.Small : ThumbnailSize.Medium;
  const defaultLibraryViewMode = osPlatform === 'android' ? LibraryViewMode.Recursive : LibraryViewMode.Flat;

  const selectedImagePathRef = useRef<string | null>(null);
  useEffect(() => {
    selectedImagePathRef.current = selectedImage?.path ?? null;
  }, [selectedImage?.path]);

  const prevAdjustmentsRef = useRef<any>(null);

  const [viewportSize, setViewportSize] = useState<ImageDimensions>(() => {
    if (typeof window === 'undefined') {
      return { width: 0, height: 0 };
    }

    return {
      width: Math.round(window.visualViewport?.width ?? window.innerWidth),
      height: Math.round(window.visualViewport?.height ?? window.innerHeight),
    };
  });

  const isBackendReadyRef = useRef(true);
  const previewJobIdRef = useRef<number>(0);
  const latestRenderedJobIdRef = useRef<number>(0);
  const currentResRef = useRef<number>(1280);
  const cachedEditStateRef = useRef<any | null>(null);

  const [libraryViewMode, setLibraryViewMode] = useState<LibraryViewMode>(defaultLibraryViewMode);
  const [isImageDragActive, setIsImageDragActive] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [thumbnailSize, setThumbnailSize] = useState(defaultThumbnailSize);
  const [thumbnailAspectRatio, setThumbnailAspectRatio] = useState(ThumbnailAspectRatio.Cover);

  const { requestThumbnails, clearThumbnailQueue, pauseThumbnailQueue, markGenerated } = useThumbnails();

  const transformWrapperRef = useRef<any>(null);
  const preloadedDataRef = useRef<{
    trees?: Promise<any>;
    images?: Promise<ImageFile[]>;
    rootPaths?: string[];
    currentPath?: string;
  }>({});

  useAppInitialization({
    preloadedDataRef,
    thumbnailSize,
    setThumbnailSize,
    thumbnailAspectRatio,
    setThumbnailAspectRatio,
    libraryViewMode,
    setLibraryViewMode,
  });

  const isAndroid = osPlatform === 'android';
  const isPortraitViewport = viewportSize.width > 0 && viewportSize.height > viewportSize.width;
  const isCompactPortrait =
    viewportSize.width > 0 && viewportSize.width <= COMPACT_EDITOR_MAX_WIDTH && isPortraitViewport;

  const compactEditorPanelMinHeight = 220;
  const compactEditorPanelMaxHeight =
    viewportSize.height > 0
      ? Math.max(compactEditorPanelMinHeight, Math.min(Math.round(viewportSize.height * 0.85), 850))
      : 520;

  const getDynamicCompactPanelHeight = () => {
    const { originalSize, adjustments } = useEditorStore.getState();
    const halfScreenHeight = viewportSize.height > 0 ? Math.round(viewportSize.height * 0.5) : 340;

    if (!selectedImage || originalSize.width === 0 || originalSize.height === 0 || viewportSize.width === 0) {
      return halfScreenHeight;
    }
    let effectiveRatio = originalSize.width / originalSize.height;
    const orientationSteps = adjustments?.orientationSteps || 0;
    if (orientationSteps % 2 !== 0) {
      effectiveRatio = originalSize.height / originalSize.width;
    }
    if (adjustments?.aspectRatio && adjustments.aspectRatio > 0) {
      effectiveRatio = adjustments.aspectRatio;
    }
    const desiredImageHeight = viewportSize.width / effectiveRatio;
    const topUiEstimation = !appSettings?.decorations && !isWindowFullScreen ? 110 : 60;
    const totalDesiredTopHeight = desiredImageHeight + topUiEstimation;
    const calculatedBottomHeight = Math.round(viewportSize.height - totalDesiredTopHeight);
    return Math.max(halfScreenHeight, calculatedBottomHeight);
  };

  const compactEditorPanelDefaultHeight = getDynamicCompactPanelHeight();
  const compactEditorPanelHeight = Math.max(
    compactEditorPanelMinHeight,
    Math.min(compactEditorPanelHeightOverride ?? compactEditorPanelDefaultHeight, compactEditorPanelMaxHeight),
  );
  const compactEditorPanelCollapsedHeight = 96;

  const { handleCopyAdjustments, handlePasteAdjustments, handleResetAdjustments, handleZoomChange } =
    useEditorActions();

  const navigationRefs = {
    transformWrapperRef,
    preloadedDataRef,
    cachedEditStateRef,
    selectedImagePathRef,
    isBackendReadyRef,
    latestRenderedJobIdRef,
    previewJobIdRef,
    currentResRef,
    prevAdjustmentsRef,
  };

  const {
    handleGoHome,
    handleBackToLibrary,
    handleImageSelect,
    handleSelectSubfolder,
    handleSelectAlbum,
    handleOpenImage,
    handlePickWorkflowImages,
    handleOpenImagePaths,
    handleOpenFolder,
    handleContinueSession,
  } = useAppNavigation({
    clearThumbnailQueue,
    refs: navigationRefs,
  });

  const handleOpenMultiImageWorkflow = useCallback(async () => {
    const title = i18n.t('modals.imageStack.title');
    const paths = await handlePickWorkflowImages(title);
    if (paths.length === 0) return;

    setUI({
      imageStackModalState: {
        detailImageBase64: null,
        error: null,
        finalImageBase64: null,
        isOpen: true,
        isProcessing: false,
        progressMessage: null,
        requestId: null,
        resultId: null,
        resultSize: null,
        sourcePaths: paths,
        blendMode: 'focus',
        alignmentMode: 'auto',
      },
    });
  }, [handlePickWorkflowImages, setUI]);

  const {
    externalEditSession,
    isFinishing: isExternalEditFinishing,
    finishExternalEdit,
  } = useExternalEditSession(handleImageSelect);

  const {
    handleRate,
    handleClearSelection,
    handleLibraryImageSingleClick,
    handleImageClick,
    handleSetColorLabel,
    refreshAllFolderTrees,
    handleTogglePinFolder,
    handleCreateAlbumItem,
    handleRenameAlbumItem,
  } = useLibraryActions(handleImageSelect);

  const { displayList: sortedImageList, badges: groupBadgeInfo } = useSortedLibrary();

  const handleLibraryRefresh = useCallback(async () => {
    if (currentFolderPath) {
      if (currentFolderPath.startsWith('Album: ')) {
        const { activeAlbumId, albumTree } = useLibraryStore.getState();
        if (activeAlbumId) {
          const findObj = (nodes: any[]): any => {
            for (const n of nodes) {
              if (n.id === activeAlbumId) return n;
              if (n.type === 'group') {
                const f = findObj(n.children);
                if (f) return f;
              }
            }
            return null;
          };
          const album = findObj(albumTree);
          if (album) await handleSelectAlbum(album.id, album.name, album.images, true);
        }
      } else {
        await handleSelectSubfolder(currentFolderPath, false, undefined, false, true);
      }
    }
  }, [currentFolderPath, handleSelectSubfolder, handleSelectAlbum]);

  const {
    executeDelete,
    handleDeleteSelected,
    handleCreateFolder,
    handleRenameFolder,
    handleSaveRename,
    handleRenameFiles,
    handleStartImport,
    handleImportClick,
    handlePasteFiles,
  } = useFileOperations(
    handleLibraryRefresh,
    refreshAllFolderTrees,
    handleImageSelect,
    handleBackToLibrary,
    sortedImageList,
  );

  const {
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
  } = useProductivityActions(handleLibraryRefresh, pauseThumbnailQueue);

  const {
    handleEditorContextMenu,
    handleThumbnailContextMenu,
    handleFolderTreeContextMenu,
    handleAlbumTreeContextMenu,
    handleMainLibraryContextMenu,
  } = useAppContextMenus({
    handleImageSelect,
    handleBackToLibrary,
    handleLibraryRefresh,
    handleRenameFiles,
    handleImportClick,
    refreshAllFolderTrees,
    refreshImageList: handleLibraryRefresh,
    executeDelete,
    handleTogglePinFolder,
  });

  useTauriListeners({
    refreshAllFolderTrees,
    handleSelectSubfolder,
    refreshImageList: handleLibraryRefresh,
    markGenerated,
  });

  useAndroidBackHandler();

  const handleToggleFullScreen = useCallback(() => {
    const { zoom, selectedImage } = useEditorStore.getState();
    const currentlyZoomed = zoom > 1.01;
    setUI({ isInstantTransition: currentlyZoomed });

    if (isFullScreen) {
      setUI({ isFullScreen: false });
    } else {
      if (!selectedImage) return;
      setUI({ isFullScreen: true });
    }

    if (currentlyZoomed) {
      setTimeout(() => setUI({ isInstantTransition: false }), 100);
    }
  }, [isFullScreen, setUI]);

  useKeyboardShortcuts({
    sortedImageList,
    handleBackToLibrary,
    handleDeleteSelected,
    handleImageSelect,
    handlePasteFiles,
    handleToggleFullScreen,
    handleZoomChange,
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const updateViewportSize = () => {
      const nextViewportSize = {
        width: Math.round(window.visualViewport?.width ?? window.innerWidth),
        height: Math.round(window.visualViewport?.height ?? window.innerHeight),
      };

      setViewportSize((prev) =>
        prev.width === nextViewportSize.width && prev.height === nextViewportSize.height ? prev : nextViewportSize,
      );
    };

    updateViewportSize();

    window.addEventListener('resize', updateViewportSize);
    window.addEventListener('orientationchange', updateViewportSize);
    window.visualViewport?.addEventListener('resize', updateViewportSize);

    return () => {
      window.removeEventListener('resize', updateViewportSize);
      window.removeEventListener('orientationchange', updateViewportSize);
      window.visualViewport?.removeEventListener('resize', updateViewportSize);
    };
  }, []);

  useEffect(() => {
    const handleGlobalContextMenu = (event: MouseEvent) => {
      event.preventDefault();
    };
    window.addEventListener('contextmenu', handleGlobalContextMenu);
    return () => window.removeEventListener('contextmenu', handleGlobalContextMenu);
  }, []);

  useEffect(() => {
    if (
      (activeRightPanel !== Panel.Masks || !activeMaskContainerId) &&
      (activeRightPanel !== Panel.Ai || !activeAiPatchContainerId)
    ) {
      setEditor({ isMaskControlHovered: false });
    }
  }, [activeRightPanel, activeMaskContainerId, activeAiPatchContainerId, setEditor]);

  const createResizeHandler = (stateKey: string, startSize: number) => (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    setIsResizing(true);

    const pointerId = e.pointerId;
    const target = e.currentTarget;
    const startX = e.clientX;
    const startY = e.clientY;

    const previousTouchAction = document.documentElement.style.touchAction;
    const previousUserSelect = document.documentElement.style.userSelect;

    target.setPointerCapture?.(pointerId);
    document.documentElement.style.touchAction = 'none';
    document.documentElement.style.userSelect = 'none';

    const doDrag = (moveEvent: PointerEvent) => {
      if (moveEvent.pointerId !== pointerId) return;
      moveEvent.preventDefault();

      if (stateKey === 'left') {
        let w = startSize + (moveEvent.clientX - startX);
        if (w < 200) w = 48;
        else if (w > 600) w = 600;
        setUI({ leftPanelWidth: Math.round(w) });
      } else if (stateKey === 'right') {
        let w = startSize - (moveEvent.clientX - startX);
        if (w < 200) w = 48;
        else if (w > 600) w = 600;
        setUI({ rightPanelWidth: Math.round(w) });
      } else if (stateKey === 'bottom') {
        const newHeight = startSize - (moveEvent.clientY - startY);
        if (newHeight < 100) {
          setUI((state) => ({
            uiVisibility: { ...state.uiVisibility, filmstrip: false },
          }));
        } else {
          setUI((state) => ({
            bottomPanelHeight: Math.round(Math.min(newHeight, 400)),
            uiVisibility: { ...state.uiVisibility, filmstrip: true },
          }));
        }
      } else if (stateKey === 'compact') {
        setUI({
          compactEditorPanelHeightOverride: Math.round(
            Math.max(
              compactEditorPanelMinHeight,
              Math.min(startSize - (moveEvent.clientY - startY), compactEditorPanelMaxHeight),
            ),
          ),
        });
      }
    };

    const stopDrag = (upEvent: PointerEvent) => {
      if (upEvent.pointerId !== pointerId) return;
      if (target.hasPointerCapture?.(pointerId)) target.releasePointerCapture(pointerId);

      document.documentElement.style.cursor = '';
      document.documentElement.style.touchAction = previousTouchAction;
      document.documentElement.style.userSelect = previousUserSelect;

      window.removeEventListener('pointermove', doDrag);
      window.removeEventListener('pointerup', stopDrag);
      window.removeEventListener('pointercancel', stopDrag);
      setIsResizing(false);
    };
    document.documentElement.style.cursor =
      stateKey === 'bottom' || stateKey === 'compact' ? 'row-resize' : 'col-resize';

    window.addEventListener('pointermove', doDrag, { passive: false });
    window.addEventListener('pointerup', stopDrag);
    window.addEventListener('pointercancel', stopDrag);
  };

  useEffect(() => {
    if (!isTauri()) return;

    const appWindow = getCurrentWindow();
    const checkFullscreen = async () => {
      setUI({ isWindowFullScreen: await appWindow.isFullscreen() });
    };
    checkFullscreen();
    const unlistenPromise = appWindow.onResized(checkFullscreen);
    return () => {
      unlistenPromise.then((unlisten: any) => unlisten());
    };
  }, [setUI]);

  useEffect(() => {
    if (!isTauri() || isAndroid) return;

    let isEffectActive = true;
    const listeners = [
      listen<{ paths: string[] }>(TauriEvent.DRAG_ENTER, () => {
        if (isEffectActive) setIsImageDragActive(true);
      }),
      listen(TauriEvent.DRAG_OVER, () => {
        if (isEffectActive) setIsImageDragActive(true);
      }),
      listen<{ paths: string[] }>(TauriEvent.DRAG_DROP, (event) => {
        if (!isEffectActive) return;
        setIsImageDragActive(false);
        void handleOpenImagePaths(event.payload.paths);
      }),
      listen(TauriEvent.DRAG_LEAVE, () => {
        if (isEffectActive) setIsImageDragActive(false);
      }),
    ];

    return () => {
      isEffectActive = false;
      listeners.forEach((listener) => {
        void listener.then((unlisten) => unlisten()).catch(console.error);
      });
    };
  }, [handleOpenImagePaths, isAndroid]);

  const handleRightPanelSelect = useCallback(
    (panelId: Panel) => {
      if (panelId === Panel.Export && activeView === 'editor' && selectedImage) {
        setUI({ isEditorExportDialogOpen: true });
        return;
      }
      setRightPanel(panelId);
      if (rightPanelWidth < 200) {
        setUI({ rightPanelWidth: 368 });
      }
      setEditor({ activeMaskId: null, activeAiSubMaskId: null, isWbPickerActive: false });
    },
    [activeView, rightPanelWidth, selectedImage, setRightPanel, setEditor, setUI],
  );

  const handleToggleFolder = useCallback(
    async (path: string) => {
      const isExpanding = !expandedFolders.has(path);
      setLibrary((state) => {
        const newSet = new Set(state.expandedFolders);
        if (isExpanding) {
          newSet.add(path);
        } else {
          newSet.delete(path);
        }
        return { expandedFolders: newSet };
      });
      if (!isExpanding) return;
      try {
        const showCounts = appSettings?.enableFolderImageCounts ?? false;
        const newChildren: any[] = await invoke(Invokes.GetFolderChildren, {
          path,
          showImageCounts: showCounts,
        });
        setLibrary((state) => ({
          folderTrees: state.folderTrees.map((t: any) => insertChildrenIntoTree(t, path, newChildren)),
        }));
        setLibrary((state) => ({
          pinnedFolderTrees: state.pinnedFolderTrees.map((tree) => insertChildrenIntoTree(tree, path, newChildren)),
        }));
      } catch (err) {
        message.error(`Failed to load folder: ${err}`);
      }
    },
    [expandedFolders, appSettings?.enableFolderImageCounts, setLibrary],
  );

  const renderAppPanel = useCallback(
    (panelId: Panel) => {
      switch (panelId) {
        case Panel.FolderTree:
          return (
            <FolderTree
              isResizing={isResizing}
              onContextMenu={handleFolderTreeContextMenu}
              onAlbumContextMenu={handleAlbumTreeContextMenu}
              onSelectAlbum={handleSelectAlbum}
              onFolderSelect={(path) => handleSelectSubfolder(path, false)}
              onToggleFolder={handleToggleFolder}
              onOpenFolder={handleOpenFolder}
              style={{ width: '100%', height: '100%' }}
              isInstantTransition={isInstantTransition}
            />
          );
        case Panel.Export:
          return (
            <ExportPanel
              exportState={exportState}
              multiSelectedPaths={multiSelectedPaths}
              selectedImage={selectedImage}
              setExportState={setExportState}
              appSettings={appSettings}
              onSettingsChange={handleSettingsChange}
              rootPaths={rootPaths}
              isVisible={true}
              onClose={() => setUI({ isLibraryExportPanelVisible: false })}
            />
          );
        case Panel.Adjustments:
          return <Controls />;
        case Panel.Metadata:
          return <MetadataPanel />;
        case Panel.Crop:
          return <CropPanel />;
        case Panel.Masks:
          return <MasksPanel />;
        case Panel.Ai:
          return BASIC_MODE ? null : <AIPanel />;
        case Panel.Presets:
          return <PresetsPanel />;
        default:
          return null;
      }
    },
    [
      isResizing,
      handleFolderTreeContextMenu,
      handleAlbumTreeContextMenu,
      handleSelectAlbum,
      handleSelectSubfolder,
      handleToggleFolder,
      handleOpenFolder,
      setUI,
      isInstantTransition,
      exportState,
      multiSelectedPaths,
      selectedImage,
      setExportState,
      appSettings,
      handleSettingsChange,
      rootPaths,
    ],
  );

  const hasRoots = rootPaths && rootPaths.length > 0;
  const hasMainContent = hasRoots || libraryImageCount > 0 || (activeView === 'editor' && !!selectedImage);
  const isDevelopWorkspace = activeView === 'editor' && !!selectedImage;

  const shouldHideFolderTree = isAndroid;
  const isWgpuActive =
    activeView === 'editor' &&
    !BASIC_MODE &&
    appSettings?.useWgpuRenderer !== false &&
    selectedImage?.isReady &&
    hasRenderedFirstFrame;
  const useMacWindowShell = osPlatform === 'macos' && !appSettings?.decorations && !isWindowFullScreen && !isFullScreen;

  const layoutSensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 5 } }));
  const handleDragStart = (e: any) => {
    if (e.active.data.current?.type === 'layout-tab') {
      setLayoutDragItem(e.active.data.current.panel as Panel);
    }
  };
  const handleDragEnd = (e: any) => {
    setLayoutDragItem(null);
    if (e.active.data.current?.type === 'layout-tab' && e.over?.data.current?.type === 'layout-region') {
      movePanel(e.active.data.current.panel as Panel, e.over.data.current.region as PanelRegion);
    }
  };
  const ActiveOverlayIcon = activeLayoutDragItem ? PANEL_ICONS[activeLayoutDragItem] : null;

  return (
    <>
      <ImageProcessingManager
        transformWrapperRef={transformWrapperRef}
        prevAdjustmentsRef={prevAdjustmentsRef}
        previewJobIdRef={previewJobIdRef}
        latestRenderedJobIdRef={latestRenderedJobIdRef}
        currentResRef={currentResRef}
      />
      <ImageLoaderManager cachedEditStateRef={cachedEditStateRef} />
      <div
        className={clsx(
          'relative flex flex-col h-screen font-sans text-text-primary overflow-hidden select-none',
          isDevelopWorkspace && 'develop-workspace-root',
          useMacWindowShell && 'macos-window-shell',
          isWgpuActive ? 'bg-transparent' : 'bg-bg-primary',
        )}
      >
        {isImageDragActive && (
          <div aria-live="polite" className="image-drop-overlay" role="status">
            <div className="image-drop-overlay-card">
              <Images aria-hidden="true" size={34} strokeWidth={1.5} />
              <strong>{i18n.t('library.splash.dropImages')}</strong>
              <span>{i18n.t('library.splash.dropImagesHint')}</span>
            </div>
          </div>
        )}
        <div
          className={clsx(
            'shrink-0 overflow-hidden z-50',
            !isInstantTransition && 'transition-[max-height,opacity] duration-300 ease-in-out',
            isFullScreen ? 'max-h-0 opacity-0 pointer-events-none' : 'max-h-[60px] opacity-100',
          )}
        >
          {appSettings?.decorations || (!isWindowFullScreen && <TitleBar />)}
        </div>
        <div
          className={clsx(
            'flex-1 flex flex-col min-h-0',
            isLayoutReady &&
              hasMainContent &&
              !isInstantTransition &&
              'transition-[padding,gap] duration-300 ease-in-out',
            [hasMainContent && (isFullScreen || isDevelopWorkspace ? 'p-0 gap-0' : 'p-2 gap-2')],
          )}
        >
          <DndContext sensors={layoutSensors} onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
            <div
              className={clsx(
                'flex flex-row grow h-full min-h-0',
                isDevelopWorkspace && 'develop-workspace',
                isDevelopWorkspace && isWgpuActive && 'is-wgpu-active',
              )}
            >
              {!shouldHideFolderTree && hasRoots && hasMainContent && !isDevelopWorkspace && (
                <SidePanelArea
                  side="left"
                  width={leftPanelWidth}
                  topRegion="leftTop"
                  bottomRegion="leftBottom"
                  renderPanel={renderAppPanel}
                  onWidthChange={createResizeHandler('left', leftPanelWidth)}
                  isResizing={isResizing}
                />
              )}
              <div className="relative flex-1 flex flex-col min-w-0">
                {selectedImage && externalEditSession && (
                  <ExternalEditBar
                    session={externalEditSession}
                    isFinishing={isExternalEditFinishing}
                    errorMessage={exportState.status === Status.Error ? exportState.errorMessage : ''}
                    onDone={finishExternalEdit}
                  />
                )}
                <div
                  className={clsx(
                    'flex-1 flex flex-col min-w-0 h-full',
                    activeView === 'editor' && selectedImage ? 'flex' : 'hidden',
                  )}
                >
                  {selectedImage && (
                    <EditorView
                      transformWrapperRef={transformWrapperRef}
                      isResizing={isResizing}
                      isCompactPortrait={isCompactPortrait}
                      isAndroid={isAndroid}
                      compactEditorPanelHeight={compactEditorPanelHeight}
                      compactEditorPanelCollapsedHeight={compactEditorPanelCollapsedHeight}
                      thumbnailAspectRatio={thumbnailAspectRatio}
                      sortedImageList={sortedImageList}
                      createResizeHandler={createResizeHandler}
                      handleBackToLibrary={handleBackToLibrary}
                      handleEditorContextMenu={handleEditorContextMenu}
                      handleThumbnailContextMenu={handleThumbnailContextMenu}
                      handleMainLibraryContextMenu={handleMainLibraryContextMenu}
                      handleImageClick={handleImageClick}
                      handleClearSelection={handleClearSelection}
                      handleCopyAdjustments={handleCopyAdjustments}
                      handlePasteAdjustments={handlePasteAdjustments}
                      handleRate={handleRate}
                      handleZoomChange={handleZoomChange}
                      handleRightPanelSelect={handleRightPanelSelect}
                      requestThumbnails={requestThumbnails}
                    />
                  )}
                </div>
                <div
                  className={clsx(
                    'flex-1 flex flex-col min-w-0 h-full',
                    activeView === 'editor' && selectedImage ? 'hidden' : 'flex',
                  )}
                >
                  <LibraryView
                    sortedImageList={sortedImageList}
                    groupBadgeInfo={groupBadgeInfo}
                    thumbnailSize={thumbnailSize}
                    thumbnailAspectRatio={thumbnailAspectRatio}
                    libraryViewMode={libraryViewMode}
                    isAndroid={isAndroid}
                    setThumbnailSize={setThumbnailSize}
                    setThumbnailAspectRatio={setThumbnailAspectRatio}
                    setLibraryViewMode={setLibraryViewMode}
                    handleClearSelection={handleClearSelection}
                    handleLibraryImageSingleClick={handleLibraryImageSingleClick}
                    handleImageSelect={handleImageSelect}
                    handleRate={handleRate}
                    handleThumbnailContextMenu={handleThumbnailContextMenu}
                    handleMainLibraryContextMenu={handleMainLibraryContextMenu}
                    handleContinueSession={handleContinueSession}
                    handleGoHome={handleGoHome}
                    handleOpenImage={handleOpenImage}
                    handleOpenMultiImageWorkflow={handleOpenMultiImageWorkflow}
                    handleOpenFolder={handleOpenFolder}
                    handleImportClick={handleImportClick}
                    handleLibraryRefresh={handleLibraryRefresh}
                    handleCopyAdjustments={handleCopyAdjustments}
                    handlePasteAdjustments={handlePasteAdjustments}
                    handleResetAdjustments={handleResetAdjustments}
                    requestThumbnails={requestThumbnails}
                  />
                </div>
                {isSettingsOpen && appSettings && hasRoots && (
                  <div className="absolute inset-0 z-50 flex bg-bg-secondary rounded-lg">
                    <div className="w-full h-full flex flex-col p-4 lg:p-8 overflow-y-auto custom-scrollbar">
                      <SettingsPanel
                        appSettings={appSettings}
                        onBack={() => setUI({ isSettingsOpen: false })}
                        onLibraryRefresh={handleLibraryRefresh}
                        onSettingsChange={handleSettingsChange}
                        rootPaths={rootPaths}
                      />
                    </div>
                  </div>
                )}
              </div>
              {!isAndroid &&
                hasMainContent &&
                (isDevelopWorkspace ? (
                  <DevelopPanel
                    activePanel={activeRightPanel}
                    isResizing={isResizing}
                    onPanelSelect={handleRightPanelSelect}
                    onWidthChange={createResizeHandler('right', rightPanelWidth)}
                    renderPanel={renderAppPanel}
                    width={rightPanelWidth}
                  />
                ) : (
                  <SidePanelArea
                    side="right"
                    width={rightPanelWidth}
                    topRegion="rightTop"
                    bottomRegion="rightBottom"
                    renderPanel={renderAppPanel}
                    onWidthChange={createResizeHandler('right', rightPanelWidth)}
                    isResizing={isResizing}
                  />
                ))}
            </div>
            <DragOverlay dropAnimation={{ duration: 120, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' }}>
              {!isDevelopWorkspace && activeLayoutDragItem && ActiveOverlayIcon ? (
                <div className="w-10 h-10 bg-surface shadow-2xl rounded-md flex items-center justify-center text-text-primary ring-1 ring-border-color">
                  <ActiveOverlayIcon size={20} />
                </div>
              ) : null}
            </DragOverlay>
          </DndContext>
        </div>
        <AppModals
          requestThumbnails={requestThumbnails}
          handleImageSelect={handleImageSelect}
          handleSavePanorama={handleSavePanorama}
          handleStartPanorama={handleStartPanorama}
          handleStartImageStack={handleStartImageStack}
          handleSaveImageStack={handleSaveImageStack}
          handleSaveHdr={handleSaveHdr}
          handleStartHdr={handleStartHdr}
          refreshImageList={handleLibraryRefresh}
          handleApplyDenoise={handleApplyDenoise}
          handleBatchDenoise={handleBatchDenoise}
          handleSaveDenoisedImage={handleSaveDenoisedImage}
          handleCreateFolder={handleCreateFolder}
          handleRenameFolder={handleRenameFolder}
          handleSaveRename={handleSaveRename}
          handleStartImport={handleStartImport}
          handleSetColorLabel={handleSetColorLabel}
          handleRate={handleRate}
          executeDelete={executeDelete}
          handleSaveCollage={handleSaveCollage}
          handleCreateAlbumItem={handleCreateAlbumItem}
          handleRenameAlbumItem={handleRenameAlbumItem}
        />
        <MessageHost topOffset={!appSettings?.decorations && !isWindowFullScreen && !isFullScreen ? 48 : 16} />
      </div>
    </>
  );
}

const AppWrapper = () => (
  <ContextMenuProvider>
    <App />
    <GlobalTooltip />
  </ContextMenuProvider>
);

export default AppWrapper;
