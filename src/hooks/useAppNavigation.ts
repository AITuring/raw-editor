import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { homeDir } from '@tauri-apps/api/path';
import { message } from '../components/ui/messageApi';
import { useLibraryStore } from '../store/useLibraryStore';
import { useEditorStore } from '../store/useEditorStore';
import { useUIStore } from '../store/useUIStore';
import { useProcessStore } from '../store/useProcessStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { Invokes, LibraryViewMode, type ImageFile, type SupportedTypes } from '../components/ui/AppProperties';
import { INITIAL_ADJUSTMENTS, normalizeLoadedAdjustments } from '../utils/adjustments';
import { globalImageCache } from '../utils/ImageLRUCache';
import { debouncedSave, debouncedSetHistory } from './useEditorActions';
import i18n from '../i18n';
import { IMAGE_STACK_MAX_SOURCES } from '../utils/imageStackPipeline';
import { computeSortedLibrary } from './useSortedLibrary';

const normalizeExtensions = (extensions: string[]) =>
  Array.from(new Set(extensions.map((extension) => extension.trim().replace(/^\./, '').toLowerCase()).filter(Boolean)));

interface FolderContentSummary {
  totalFiles: number;
  supportedFiles: number;
  unreadableEntries: number;
}

const loadSupportedImageTypes = async (): Promise<SupportedTypes> => {
  const settingsStore = useSettingsStore.getState();
  if (settingsStore.supportedTypes) return settingsStore.supportedTypes;

  const supportedTypes = await invoke<SupportedTypes>(Invokes.GetSupportedFileTypes);
  settingsStore.setSupportedTypes(supportedTypes);
  return supportedTypes;
};

export interface AppNavigationProps {
  clearThumbnailQueue: () => void;
  refs: {
    transformWrapperRef: React.RefObject<any>;
    cachedEditStateRef: React.RefObject<any>;
    selectedImagePathRef: React.RefObject<string | null>;
    isBackendReadyRef: React.RefObject<boolean>;
    latestRenderedJobIdRef: React.RefObject<number>;
    previewJobIdRef: React.RefObject<number>;
    currentResRef: React.RefObject<number>;
    prevAdjustmentsRef: React.RefObject<any>;
  };
}

export function useAppNavigation({ clearThumbnailQueue, refs }: AppNavigationProps) {
  const {
    transformWrapperRef,
    cachedEditStateRef,
    selectedImagePathRef,
    isBackendReadyRef,
    latestRenderedJobIdRef,
    previewJobIdRef,
    currentResRef,
    prevAdjustmentsRef,
  } = refs;

  const handleGoHome = useCallback(() => {
    useLibraryStore.getState().setLibrary({
      rootPaths: [],
      currentFolderPath: null,
      activeAlbumId: null,
      imageList: [],
      imageRatings: {},
      folderTrees: [],
      multiSelectedPaths: [],
      libraryActivePath: null,
      expandedFolders: new Set(),
      contentState: {
        status: 'idle',
        error: null,
        totalFiles: 0,
        supportedFiles: 0,
        unreadableEntries: 0,
      },
    });
    useUIStore.getState().setUI({
      isEditorExportDialogOpen: false,
      libraryContextPanel: null,
      isLibraryQuickPreviewOpen: false,
    });
  }, []);

  const handleBackToLibrary = useCallback(() => {
    const { selectedImage } = useEditorStore.getState();
    const { setLibrary } = useLibraryStore.getState();
    const { setUI } = useUIStore.getState();

    if (selectedImage?.path && cachedEditStateRef.current) {
      globalImageCache.set(selectedImage.path, cachedEditStateRef.current);
    }
    if (transformWrapperRef.current) {
      transformWrapperRef.current.resetTransform(0);
    }
    useEditorStore.getState().setEditor({ zoom: 1 });

    debouncedSave.flush();
    debouncedSetHistory.cancel();

    const lastActivePath = selectedImage?.path ?? null;

    setLibrary({ libraryActivePath: lastActivePath });
    setUI({
      activeView: 'library',
      isEditorExportDialogOpen: false,
      libraryContextPanel: null,
      isLibraryQuickPreviewOpen: false,
      slideDirection: 1,
    });
  }, [refs]);

  const handleImageSelect = useCallback(
    async (path: string, openInEditor: boolean = true) => {
      const { selectedImage, isSliderDragging, resetHistory, setEditor } = useEditorStore.getState();
      const { setLibrary, multiSelectedPaths } = useLibraryStore.getState();
      const { setUI } = useUIStore.getState();

      if (openInEditor) {
        setUI({ activeView: 'editor', libraryContextPanel: null, isLibraryQuickPreviewOpen: false });
      }

      if (selectedImage?.path === path) {
        if (!selectedImage.isReady) {
          if (openInEditor) setLibrary({ isViewLoading: true });
          setEditor((state) => ({
            imageLoadError: null,
            imageLoadRevision: state.imageLoadRevision + 1,
          }));
        }
        return;
      }

      useEditorStore.getState().patchesSentToBackend.clear();
      debouncedSave.flush();
      debouncedSetHistory.cancel();

      if (selectedImage?.path && cachedEditStateRef.current) {
        globalImageCache.set(selectedImage.path, cachedEditStateRef.current);
      }

      const cached = globalImageCache.get(path);
      const isFrontendCached = Boolean(cached && cached.selectedImage?.isReady);
      // A ready flag belongs to the currently displayed native texture, not to
      // the next selected path. Keep the target thumbnail/cached JPEG visible;
      // full mode may replace it only after WGPU confirms that exact image.
      setEditor({ hasRenderedFirstFrame: false });

      selectedImagePathRef.current = path;

      const newMultiSelectedPaths = multiSelectedPaths.includes(path) ? multiSelectedPaths : [path];

      setLibrary({
        multiSelectedPaths: newMultiSelectedPaths,
        libraryActivePath: path,
        selectionAnchorPath: path,
      });

      setEditor({
        showOriginal: false,
        imageLoadError: null,
        activeMaskId: null,
        activeMaskContainerId: null,
        activeAiPatchContainerId: null,
        activeAiSubMaskId: null,
        isWbPickerActive: false,
        transformedOriginalUrl: null,
      });

      setUI({
        ...(openInEditor ? { libraryContextPanel: null, isLibraryQuickPreviewOpen: false } : {}),
        compactEditorPanelHeightOverride: null,
      });

      if (isFrontendCached && cached) {
        setEditor({
          selectedImage: {
            ...cached.selectedImage,
            thumbnailUrl: useProcessStore.getState().thumbnails[path] || cached.selectedImage.thumbnailUrl,
          },
          originalSize: cached.originalSize,
          previewSize: cached.previewSize,
          histogram: cached.histogram,
          waveform: cached.waveform,
          finalPreviewUrl: cached.finalPreviewUrl,
          uncroppedAdjustedPreviewUrl: cached.uncroppedPreviewUrl,
        });

        setEditor({ adjustments: cached.adjustments });
        resetHistory(cached.adjustments);
        prevAdjustmentsRef.current = { path, adjustments: cached.adjustments };

        setLibrary({ isViewLoading: false });

        latestRenderedJobIdRef.current = previewJobIdRef.current;
        isBackendReadyRef.current = false;
        currentResRef.current = Infinity;

        invoke(Invokes.LoadImage, { path })
          .then((_result: any) => {
            if (selectedImagePathRef.current !== path) return;
            isBackendReadyRef.current = true;
            currentResRef.current = 0;
            setEditor({ originalSize: { width: _result.width, height: _result.height } });
          })
          .catch((err: any) => {
            if (String(err).includes('cancelled')) return;
            console.error('Background load_image failed on cache hit:', err);
            isBackendReadyRef.current = true;
            currentResRef.current = 0;
          });

        invoke(Invokes.LoadMetadata, { path })
          .then((metadata: any) => {
            if (selectedImagePathRef.current !== path) return;
            let freshAdjustments: any;
            if (metadata.adjustments && !metadata.adjustments.is_null) {
              freshAdjustments = normalizeLoadedAdjustments(metadata.adjustments);
            } else {
              freshAdjustments = { ...INITIAL_ADJUSTMENTS };
            }
            if (!isSliderDragging && JSON.stringify(cached.adjustments) !== JSON.stringify(freshAdjustments)) {
              setEditor({ adjustments: freshAdjustments });
              resetHistory(freshAdjustments);
              prevAdjustmentsRef.current = { path, adjustments: freshAdjustments };
              globalImageCache.set(path, { ...cached, adjustments: freshAdjustments });
            }
          })
          .catch((err) => console.error('Failed background metadata sync on cache hit:', err));

        return;
      }

      isBackendReadyRef.current = true;

      const imageFile = useLibraryStore.getState().imageList.find((img) => img.path === path);
      setEditor({
        selectedImage: {
          exif: null,
          group_id: imageFile?.group_id ?? null,
          height: 0,
          isRaw: false,
          isReady: false,
          metadata: null,
          originalUrl: null,
          path,
          thumbnailUrl: useProcessStore.getState().thumbnails[path],
          width: 0,
        },
        originalSize: { width: 0, height: 0 },
        previewSize: { width: 0, height: 0 },
        histogram: null,
        waveform: null,
        uncroppedAdjustedPreviewUrl: null,
        imageLoadError: null,
      });

      if (openInEditor) setLibrary({ isViewLoading: true });

      setEditor((state) => {
        const prev = state.finalPreviewUrl;
        if (prev?.startsWith('blob:') && !globalImageCache.isProtected(prev)) {
          setTimeout(() => {
            if (!globalImageCache.isProtected(prev)) {
              URL.revokeObjectURL(prev);
            }
          }, 250);
        }
        return { finalPreviewUrl: null };
      });

      setEditor((state) => {
        if (state.interactivePatch?.url) URL.revokeObjectURL(state.interactivePatch.url);
        return { interactivePatch: null };
      });
    },
    [refs],
  );

  const handleSelectSubfolder = useCallback(
    async (
      path: string | null,
      isNewRoot = false,
      preloadedImages?: ImageFile[],
      expandParents = true,
      preserveEditor = false,
    ) => {
      const { appSettings, handleSettingsChange } = useSettingsStore.getState();
      const { pinnedFolders } = appSettings || { pinnedFolders: [] };
      const { setLibrary, sortCriteria } = useLibraryStore.getState();
      const { setUI } = useUIStore.getState();
      const { setProcess } = useProcessStore.getState();
      const { selectedImage, resetHistory, setEditor } = useEditorStore.getState();
      const libraryViewMode = appSettings?.libraryViewMode;

      if (!preserveEditor) {
        await invoke('cancel_thumbnail_generation');
        clearThumbnailQueue();
        setLibrary({
          isViewLoading: true,
          activeAlbumId: null,
          libraryScrollTop: 0,
          contentState: {
            status: 'loading',
            error: null,
            totalFiles: 0,
            supportedFiles: 0,
            unreadableEntries: 0,
          },
        });
        useLibraryStore.getState().setSearchCriteria({ tags: [], text: '', mode: 'OR' });
        setProcess({ thumbnails: {} });
        globalImageCache.clear();
        setUI({
          activeView: 'library',
          isEditorExportDialogOpen: false,
          libraryContextPanel: null,
          isLibraryQuickPreviewOpen: false,
        });
      } else {
        setLibrary({
          isViewLoading: true,
          contentState: {
            status: 'loading',
            error: null,
            totalFiles: 0,
            supportedFiles: 0,
            unreadableEntries: 0,
          },
        });
      }

      try {
        const { rootPaths, expandedFolders: currentExpandedFolders } = useLibraryStore.getState();
        let newExpandedFolders = new Set(currentExpandedFolders);

        if (isNewRoot && path) {
          newExpandedFolders = new Set([path]);
          if (appSettings) {
            handleSettingsChange({ ...appSettings, lastRootPath: path } as any);
          }
        } else if (path && expandParents) {
          const allRoots = [...(rootPaths || []), ...(pinnedFolders || [])].filter(Boolean) as string[];
          const relevantRoot = allRoots.find((r) => path.startsWith(r));

          if (relevantRoot) {
            const separator = path.includes('/') ? '/' : '\\';
            const parentSeparatorIndex = path.lastIndexOf(separator);

            if (parentSeparatorIndex > -1 && path.length > relevantRoot.length) {
              let current = path.substring(0, parentSeparatorIndex);
              while (current && current.length >= relevantRoot.length) {
                newExpandedFolders.add(current);
                const nextParentIndex = current.lastIndexOf(separator);
                if (nextParentIndex === -1 || current === relevantRoot) break;
                current = current.substring(0, nextParentIndex);
              }
            }
            newExpandedFolders.add(relevantRoot);
          }
        }

        setLibrary({
          currentFolderPath: path,
          expandedFolders: newExpandedFolders,
          ...(preserveEditor ? {} : { imageList: [], multiSelectedPaths: [], libraryActivePath: null }),
        });

        if (!preserveEditor && selectedImage) {
          debouncedSave.flush();
          debouncedSetHistory.cancel();
          setEditor({ selectedImage: null, finalPreviewUrl: null, uncroppedAdjustedPreviewUrl: null, histogram: null });
          setEditor({ adjustments: INITIAL_ADJUSTMENTS });
          resetHistory(INITIAL_ADJUSTMENTS);
          useEditorStore.getState().patchesSentToBackend.clear();
        }

        const command =
          libraryViewMode === LibraryViewMode.Recursive ? Invokes.ListImagesRecursive : Invokes.ListImagesInDir;

        const summaryPromise = path
          ? invoke<FolderContentSummary>(Invokes.InspectFolderContents, {
              path,
              recursive: libraryViewMode === LibraryViewMode.Recursive,
            }).catch((error) => {
              console.warn('Failed to inspect folder contents:', error);
              return null;
            })
          : Promise.resolve(null);

        const files: ImageFile[] = preloadedImages ?? (await invoke(command, { path }));
        const summary = await summaryPromise;

        const contentStatus = files.length > 0 ? 'ready' : summary && summary.totalFiles > 0 ? 'unsupported' : 'empty';

        setLibrary({
          contentState: {
            status: contentStatus,
            error: null,
            totalFiles: summary?.totalFiles ?? files.length,
            supportedFiles: summary?.supportedFiles ?? files.length,
            unreadableEntries: summary?.unreadableEntries ?? 0,
          },
        });

        const initialRatings: Record<string, number> = {};
        files.forEach((f) => {
          if (f.rating !== undefined) {
            initialRatings[f.path] = f.rating;
          }
        });
        setLibrary({ imageRatings: initialRatings });

        const exifSortKeys = ['date_taken', 'iso', 'shutter_speed', 'aperture', 'focal_length'];
        const isExifSortActive = exifSortKeys.includes(sortCriteria.key);

        if (files.length > 0) {
          const paths = files.map((f: ImageFile) => f.path);

          if (isExifSortActive) {
            let combinedExifMap: Record<string, any> = {};
            const chunkSize = 100;

            for (let i = 0; i < paths.length; i += chunkSize) {
              const chunk = paths.slice(i, i + chunkSize);
              try {
                const chunkExif: any = await invoke(Invokes.ReadExifForPaths, { paths: chunk });
                combinedExifMap = { ...combinedExifMap, ...chunkExif };
              } catch (err) {
                console.error('Failed to read EXIF chunk:', err);
              }
            }

            const finalImageList = files.map((image) => ({
              ...image,
              exif: combinedExifMap[image.path] || image.exif || null,
            }));
            setLibrary({ imageList: finalImageList });
          } else {
            setLibrary({ imageList: files });

            setTimeout(() => {
              const fetchExifInChunks = async () => {
                const chunkSize = 50;
                for (let i = 0; i < paths.length; i += chunkSize) {
                  if (useLibraryStore.getState().currentFolderPath !== path) break;

                  const chunk = paths.slice(i, i + chunkSize);
                  try {
                    const chunkExif: any = await invoke(Invokes.ReadExifForPaths, { paths: chunk });
                    setLibrary((state) => ({
                      imageList: state.imageList.map((image) => ({
                        ...image,
                        exif: chunkExif[image.path] || image.exif || null,
                      })),
                    }));
                    await new Promise((resolve) => setTimeout(resolve, 50));
                  } catch (err) {
                    console.error('Failed to read EXIF chunk:', err);
                  }
                }
              };
              fetchExifInChunks();
            }, 500);
          }
        } else {
          setLibrary({ imageList: files });
        }

        if (!preserveEditor && files.length > 0) {
          const firstVisibleImage = computeSortedLibrary(useLibraryStore.getState(), useSettingsStore.getState())[0];
          const firstPath = firstVisibleImage?.path ?? null;
          setLibrary({
            libraryActivePath: firstPath,
            multiSelectedPaths: firstPath ? [firstPath] : [],
            selectionAnchorPath: firstPath,
          });
        }

        if (!preserveEditor) {
          invoke(Invokes.StartBackgroundIndexing, { folderPath: path }).catch((err) => {
            console.error('Failed to start background indexing:', err);
          });
        }
      } catch (err) {
        console.error('Failed to load folder contents:', err);
        message.error('Failed to load images from the selected folder.');
        setLibrary({
          imageList: [],
          multiSelectedPaths: [],
          libraryActivePath: null,
          selectionAnchorPath: null,
          contentState: {
            status: 'error',
            error: String(err),
            totalFiles: 0,
            supportedFiles: 0,
            unreadableEntries: 0,
          },
        });
      } finally {
        useLibraryStore.getState().setLibrary({ isViewLoading: false });
      }
    },
    [clearThumbnailQueue, refs],
  );

  const handleSelectAlbum = useCallback(
    async (albumId: string, albumName: string, imagePaths: string[], preserveEditor = false) => {
      const { setLibrary } = useLibraryStore.getState();
      const { setUI } = useUIStore.getState();

      if (!preserveEditor) {
        await invoke('cancel_thumbnail_generation');
        clearThumbnailQueue();
        useLibraryStore.getState().setSearchCriteria({ tags: [], text: '', mode: 'OR' });
        setLibrary({ libraryScrollTop: 0 });
        globalImageCache.clear();
        setUI({
          activeView: 'library',
          isEditorExportDialogOpen: false,
          libraryContextPanel: null,
          isLibraryQuickPreviewOpen: false,
        });
      }

      setLibrary({
        isViewLoading: true,
        currentFolderPath: `Album: ${albumName}`,
        activeAlbumId: albumId,
        contentState: {
          status: 'loading',
          error: null,
          totalFiles: 0,
          supportedFiles: 0,
          unreadableEntries: 0,
        },
      });

      try {
        const files: ImageFile[] = await invoke(Invokes.GetAlbumImages, { paths: imagePaths });

        const initialRatings: Record<string, number> = {};
        files.forEach((f) => {
          if (f.rating !== undefined) initialRatings[f.path] = f.rating;
        });

        setLibrary({
          imageList: files,
          imageRatings: initialRatings,
          contentState: {
            status: files.length > 0 ? 'ready' : 'empty',
            error: null,
            totalFiles: files.length,
            supportedFiles: files.length,
            unreadableEntries: 0,
          },
          ...(preserveEditor ? {} : { multiSelectedPaths: [], libraryActivePath: null, selectionAnchorPath: null }),
        });
        if (!preserveEditor && files.length > 0) {
          const firstVisibleImage = computeSortedLibrary(useLibraryStore.getState(), useSettingsStore.getState())[0];
          const firstPath = firstVisibleImage?.path ?? null;
          setLibrary({
            multiSelectedPaths: firstPath ? [firstPath] : [],
            libraryActivePath: firstPath,
            selectionAnchorPath: firstPath,
          });
        }
      } catch (err) {
        console.error('Failed to load album images:', err);
        message.error(`Failed to load album: ${err}`);
        setLibrary({
          imageList: [],
          contentState: {
            status: 'error',
            error: String(err),
            totalFiles: 0,
            supportedFiles: 0,
            unreadableEntries: 0,
          },
        });
      } finally {
        setLibrary({ isViewLoading: false });
      }
    },
    [clearThumbnailQueue],
  );

  const handleOpenImagePaths = useCallback(
    async (paths: string[]) => {
      const uniquePaths = Array.from(new Set(paths.filter((path) => typeof path === 'string' && path.trim())));
      if (uniquePaths.length === 0) return;

      try {
        const supportedTypes = await loadSupportedImageTypes();
        const allowedExtensions = new Set(normalizeExtensions([...supportedTypes.raw, ...supportedTypes.nonRaw]));
        const validPaths = uniquePaths.filter((path) => {
          const physicalPath = path.split('?vc=')[0];
          const extension = physicalPath.split(/[\\/]/).pop()?.split('.').pop()?.toLowerCase() ?? '';
          return allowedExtensions.has(extension);
        });

        if (validPaths.length === 0) {
          message.error(i18n.t('library.import.noSupportedFiles'));
          return;
        }
        if (validPaths.length !== uniquePaths.length) {
          message.info(
            i18n.t('library.import.skippedUnsupported', {
              skipped: uniquePaths.length - validPaths.length,
            }),
          );
        }

        await invoke('cancel_thumbnail_generation');
        clearThumbnailQueue();
        useLibraryStore.getState().setSearchCriteria({ tags: [], text: '', mode: 'OR' });
        useLibraryStore.getState().setLibrary({ isViewLoading: true });
        useUIStore.getState().setUI({
          activeView: 'library',
          isEditorExportDialogOpen: false,
          libraryContextPanel: null,
          isLibraryQuickPreviewOpen: false,
        });

        const files = await invoke<ImageFile[]>(Invokes.GetAlbumImages, { paths: validPaths });
        if (files.length === 0) {
          message.error(i18n.t('library.import.noSupportedFiles'));
          return;
        }

        const imageRatings = Object.fromEntries(files.map((file) => [file.path, file.rating ?? 0]));
        const firstPath = files[0].path;
        useLibraryStore.getState().setLibrary({
          activeAlbumId: null,
          currentFolderPath: null,
          imageList: files,
          imageRatings,
          libraryActivePath: firstPath,
          libraryScrollTop: 0,
          multiSelectedPaths: [firstPath],
          selectionAnchorPath: firstPath,
          contentState: {
            status: 'ready',
            error: null,
            totalFiles: files.length,
            supportedFiles: files.length,
            unreadableEntries: 0,
          },
        });

        await handleImageSelect(firstPath);
      } catch (error) {
        console.error('Failed to open selected image files:', error);
        message.error(i18n.t('library.import.openFailed'));
      } finally {
        useLibraryStore.getState().setLibrary({ isViewLoading: false });
      }
    },
    [clearThumbnailQueue, handleImageSelect],
  );

  const handleOpenImage = useCallback(async () => {
    try {
      const supportedTypes = await loadSupportedImageTypes();
      const rawExtensions = normalizeExtensions(supportedTypes.raw);
      const nonRawExtensions = normalizeExtensions(supportedTypes.nonRaw);
      const allExtensions = Array.from(new Set([...rawExtensions, ...nonRawExtensions]));
      const selected = await open({
        defaultPath: await homeDir(),
        filters: [
          { name: i18n.t('library.import.allSupportedImages'), extensions: allExtensions },
          { name: 'RAW', extensions: rawExtensions },
          { name: 'JPEG / PNG / TIFF', extensions: nonRawExtensions },
        ],
        multiple: false,
        title: i18n.t('library.splash.openImage'),
      });

      if (typeof selected === 'string') {
        await handleOpenImagePaths([selected]);
      }
    } catch (error) {
      console.error('Failed to open image dialog:', error);
      message.error(i18n.t('library.import.openFailed'));
    }
  }, [handleOpenImagePaths]);

  const handlePickWorkflowImages = useCallback(async (title: string): Promise<string[]> => {
    try {
      const supportedTypes = await loadSupportedImageTypes();
      const rawExtensions = normalizeExtensions(supportedTypes.raw);
      const nonRawExtensions = normalizeExtensions(supportedTypes.nonRaw);
      const allExtensions = Array.from(new Set([...rawExtensions, ...nonRawExtensions]));
      const selected = await open({
        defaultPath: await homeDir(),
        filters: [
          { name: i18n.t('library.import.allSupportedImages'), extensions: allExtensions },
          { name: 'RAW', extensions: rawExtensions },
          { name: 'JPEG / PNG / TIFF', extensions: nonRawExtensions },
        ],
        multiple: true,
        title,
      });

      const paths = Array.isArray(selected) ? selected : typeof selected === 'string' ? [selected] : [];
      if (paths.length === 0) return [];

      if (paths.length < 2 || paths.length > IMAGE_STACK_MAX_SOURCES) {
        message.info(i18n.t('library.splash.multiImageSelectionHint'));
        return [];
      }

      return paths;
    } catch (error) {
      console.error('Failed to open multi-image workflow dialog:', error);
      message.error(i18n.t('library.import.openFailed'));
      return [];
    }
  }, []);

  const handleOpenFolder = async () => {
    const { osPlatform, appSettings, handleSettingsChange } = useSettingsStore.getState();
    const { rootPaths, folderTrees, setLibrary } = useLibraryStore.getState();
    const isAndroid = osPlatform === 'android';

    try {
      let selectedPath = '';
      if (isAndroid) {
        selectedPath = await invoke<string>(Invokes.GetOrCreateInternalLibraryRoot);
      } else {
        const selected = await open({ directory: true, multiple: false, defaultPath: await homeDir() });
        if (typeof selected === 'string') {
          selectedPath = selected;
        }
      }

      if (selectedPath) {
        if (!rootPaths.includes(selectedPath)) {
          const newRootPaths = [...rootPaths, selectedPath];
          setLibrary({ rootPaths: newRootPaths });

          if (appSettings) {
            handleSettingsChange({ ...appSettings, rootFolders: newRootPaths } as any);
          }

          setLibrary({ isTreeLoading: true });
          try {
            const newTree = await invoke(Invokes.GetFolderTree, {
              path: selectedPath,
              expandedFolders: [selectedPath],
              showImageCounts:
                appSettings?.enableFolderImageCounts || appSettings?.folderTreeSort?.key === 'imageCount',
            });
            setLibrary({ folderTrees: [...folderTrees, newTree] });
          } catch (e) {
            message.error(`Failed to load folder tree: ${e}`);
          } finally {
            setLibrary({ isTreeLoading: false });
          }
        }
        await handleSelectSubfolder(selectedPath, true);
      }
    } catch (err) {
      console.error(isAndroid ? 'Failed to open Android library root:' : 'Failed to open directory dialog:', err);
      message.error(isAndroid ? 'Failed to open library.' : 'Failed to open folder selection dialog.');
    }
  };

  return {
    handleGoHome,
    handleBackToLibrary,
    handleImageSelect,
    handleSelectSubfolder,
    handleSelectAlbum,
    handleOpenImage,
    handlePickWorkflowImages,
    handleOpenImagePaths,
    handleOpenFolder,
  };
}
