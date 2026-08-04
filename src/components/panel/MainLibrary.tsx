import { useState, useEffect, useMemo } from 'react';
import { getVersion } from '@tauri-apps/api/app';
import {
  AlertTriangle,
  Check,
  Folder,
  FolderInput,
  Home,
  ImagePlus,
  Loader2,
  RefreshCw,
  Settings,
  Search,
  LayoutGrid,
  Columns,
  SlidersHorizontal,
  Rows3,
} from 'lucide-react';
import CullingView from './library/CullingView';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import {
  AppSettings,
  ImageFile,
  LibraryViewMode,
  Progress,
  ThumbnailSize,
  ThumbnailAspectRatio,
  RawStatus,
  EditedStatus,
  LibraryDisplayMode,
} from '../ui/AppProperties';
import { GroupBadgeInfo, GroupId } from '../../utils/imageGrouping';
import { ImportState, Status } from '../ui/ExportImportProperties';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';
import SettingsPanel from './SettingsPanel';
import { COVER_IMAGES, COVER_ROTATION_INTERVAL_MS } from '../../config/coverImages';

import LibraryGrid from './library/LibraryGrid';
import { SearchInput, ViewOptionsDropdown } from './library/LibraryHeader';

export interface ColumnWidths {
  thumbnail: number;
  name: number;
  date: number;
  rating: number;
  color: number;
}

interface MainLibraryProps {
  activePath: string | null;
  aiModelDownloadStatus: string | null;
  appSettings: AppSettings | null;
  currentFolderPath: string | null;
  groupBadgeInfo: Map<GroupId, GroupBadgeInfo> | null;
  imageList: Array<ImageFile>;
  imageRatings: Record<string, number>;
  importState: ImportState;
  indexingProgress: Progress;
  isLoading: boolean;
  isIndexing: boolean;
  isAndroid: boolean;
  isTreeLoading: boolean;
  libraryViewMode: LibraryViewMode;
  multiSelectedPaths: Array<string>;
  onClearSelection(): void;
  onContextMenu(event: any, path: string): void;
  onContinueSession(): void;
  onEmptyAreaContextMenu(event: any): void;
  onGoHome(): void;
  onImageClick(path: string, event: any): void;
  onImageDoubleClick(path: string): void;
  onImportClick(): void;
  onLibraryRefresh(): void;
  onOpenImage(): void;
  onOpenFolder(): void;
  onSettingsChange(settings: AppSettings): Promise<void>;
  onThumbnailAspectRatioChange(aspectRatio: ThumbnailAspectRatio): void;
  onThumbnailSizeChange(size: ThumbnailSize): void;
  onRequestThumbnails?(paths: string[]): void;
  rootPaths: string[];
  setLibraryViewMode(mode: LibraryViewMode): void;
  thumbnailAspectRatio: ThumbnailAspectRatio;
  thumbnailProgress: Progress;
  thumbnailSize: ThumbnailSize;
}

export interface ColumnWidths {
  thumbnail: number;
  name: number;
  date: number;
  rating: number;
  color: number;
  shutter: number;
  aperture: number;
  iso: number;
  focal: number;
}

interface DisplayModeSwitchProps {
  displayMode: LibraryDisplayMode;
  setDisplayMode: (mode: LibraryDisplayMode) => void;
  t: any;
}

function DisplayModeSwitch({ displayMode, setDisplayMode, t }: DisplayModeSwitchProps) {
  const prefersReducedMotion = useReducedMotion();
  const options = useMemo(
    () => [
      {
        id: LibraryDisplayMode.Grid,
        Icon: LayoutGrid,
        tooltip: t('library.viewMode.grid', { defaultValue: 'Grid View' }),
      },
      {
        id: LibraryDisplayMode.List,
        Icon: Rows3,
        tooltip: t('library.viewMode.list', { defaultValue: 'List View' }),
      },
      {
        id: LibraryDisplayMode.Cull,
        Icon: Columns,
        tooltip: t('library.viewMode.culling', { defaultValue: 'Culling View' }),
      },
    ],
    [t],
  );

  const selectedIndex = options.findIndex((opt) => opt.id === displayMode);
  const safeIndex = selectedIndex >= 0 ? selectedIndex : 0;

  return (
    <div className="flex items-center bg-surface p-1 rounded-lg border border-border-color/20 h-14 w-40 select-none">
      <div className="relative flex w-full h-full">
        <motion.div
          className="absolute top-0 bottom-0 z-0 bg-bg-primary rounded-md shadow-sm"
          initial={false}
          animate={{
            x: `${safeIndex * 100}%`,
            width: `${100 / options.length}%`,
          }}
          transition={prefersReducedMotion ? { duration: 0 } : { type: 'spring', bounce: 0.2, duration: 0.6 }}
        />
        {options.map((opt) => {
          const Icon = opt.Icon;
          const isActive = displayMode === opt.id;
          return (
            <button
              aria-label={opt.tooltip}
              aria-pressed={isActive}
              key={opt.id}
              onClick={() => setDisplayMode(opt.id)}
              className={`relative z-10 flex-1 h-full flex items-center justify-center rounded-md transition-colors duration-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-1 ${
                isActive ? 'text-text-primary' : 'text-text-secondary hover:text-text-primary'
              }`}
              data-tooltip={opt.tooltip}
              style={{ WebkitTapHighlightColor: 'transparent' }}
              type="button"
            >
              <Icon aria-hidden="true" className="w-5 h-5" />
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default function MainLibrary(props: MainLibraryProps) {
  const { t } = useTranslation();
  const prefersReducedMotion = useReducedMotion();
  const setUI = useUIStore((state) => state.setUI);
  const [coverImageIndex, setCoverImageIndex] = useState(() =>
    COVER_IMAGES.length > 0 ? Math.floor(Math.random() * COVER_IMAGES.length) : 0,
  );
  const [appVersion, setAppVersion] = useState('');
  const [isBusyDelayed, setIsBusyDelayed] = useState(false);
  const [isBusyLoaderMounted, setIsBusyLoaderMounted] = useState(false);
  const [isProgressHovered, setIsProgressHovered] = useState(false);
  const isSettingsOpen = useUIStore((state) => state.isSettingsOpen);

  const libraryDisplayMode = props.appSettings?.libraryDisplayMode || LibraryDisplayMode.Grid;

  const setLibraryDisplayMode = (mode: LibraryDisplayMode) => {
    if (props.appSettings) {
      props.onSettingsChange({
        ...props.appSettings,
        libraryDisplayMode: mode,
      });
    }
  };

  const searchCriteria = useLibraryStore((state) => state.searchCriteria);

  const translatedRatingFilterOptions = useMemo(
    () => [
      { value: 0, label: t('library.filters.rating.all') },
      { value: -1, label: t('library.filters.rating.unrated') },
      { value: 1, label: t('library.filters.rating.oneAndUp') },
      { value: 2, label: t('library.filters.rating.twoAndUp') },
      { value: 3, label: t('library.filters.rating.threeAndUp') },
      { value: 4, label: t('library.filters.rating.fourAndUp') },
      { value: 5, label: t('library.filters.rating.fiveOnly') },
    ],
    [t],
  );

  const translatedRawStatusOptions = useMemo(
    () => [
      { key: RawStatus.All, label: t('library.filters.raw.all') },
      { key: RawStatus.RawOnly, label: t('library.filters.raw.rawOnly') },
      { key: RawStatus.NonRawOnly, label: t('library.filters.raw.nonRawOnly') },
    ],
    [t],
  );

  const translatedEditedStatusOptions = useMemo(
    () => [
      { key: EditedStatus.All, label: t('library.filters.edited.all') },
      { key: EditedStatus.EditedOnly, label: t('library.filters.edited.editedOnly') },
      { key: EditedStatus.UneditedOnly, label: t('library.filters.edited.uneditedOnly') },
    ],
    [t],
  );

  const translatedThumbnailSizeOptions = useMemo(
    () => [
      { id: ThumbnailSize.Small, label: t('library.thumbnailSize.small'), size: 160 },
      { id: ThumbnailSize.Medium, label: t('library.thumbnailSize.medium'), size: 240 },
      { id: ThumbnailSize.Large, label: t('library.thumbnailSize.large'), size: 320 },
    ],
    [t],
  );

  const translatedThumbnailAspectRatioOptions = useMemo(
    () => [
      { id: ThumbnailAspectRatio.Cover, label: t('library.thumbnailFit.fillSquare') },
      { id: ThumbnailAspectRatio.Contain, label: t('library.thumbnailFit.originalRatio') },
    ],
    [t],
  );

  const translatedSortOptions = useMemo(
    () => [
      { key: 'name', label: t('library.sort.fileName') },
      { key: 'date', label: t('library.sort.dateModified') },
      { key: 'rating', label: t('library.sort.rating') },
      { key: 'date_taken', label: t('library.sort.dateTaken') },
      { key: 'focal_length', label: t('library.sort.focalLength') },
      { key: 'iso', label: t('library.sort.iso') },
      { key: 'shutter_speed', label: t('library.sort.shutterSpeed') },
      { key: 'aperture', label: t('library.sort.aperture') },
      { key: 'edited', label: t('library.sort.editedStatus') },
    ],
    [t],
  );

  const isBusy =
    props.isLoading ||
    ((props.thumbnailProgress?.total ?? 0) > 0 &&
      (props.thumbnailProgress?.current ?? 0) < (props.thumbnailProgress?.total ?? 0));

  useEffect(() => {
    let timer: number | undefined;

    if (isBusy) {
      timer = window.setTimeout(() => setIsBusyDelayed(true), 1000);
    } else {
      timer = window.setTimeout(() => setIsBusyDelayed(false), 500);
    }

    return () => clearTimeout(timer);
  }, [isBusy]);

  useEffect(() => {
    if (isBusyDelayed) {
      setIsBusyLoaderMounted(true);
    }
  }, [isBusyDelayed]);

  useEffect(() => {
    getVersion()
      .then(setAppVersion)
      .catch(() => setAppVersion(''));
  }, []);

  const isCoverVisible = (!props.rootPaths || props.rootPaths.length === 0) && props.imageList.length === 0;
  useEffect(() => {
    if (!isCoverVisible || prefersReducedMotion || COVER_IMAGES.length < 2) return;

    const rotationTimer = window.setInterval(() => {
      setCoverImageIndex((currentIndex) => (currentIndex + 1) % COVER_IMAGES.length);
    }, COVER_ROTATION_INTERVAL_MS);

    return () => window.clearInterval(rotationTimer);
  }, [isCoverVisible, prefersReducedMotion]);

  useEffect(() => {
    if (!isCoverVisible || COVER_IMAGES.length < 2) return;
    const nextCoverImage = COVER_IMAGES[(coverImageIndex + 1) % COVER_IMAGES.length];
    if (!nextCoverImage) return;

    const nextImage = new Image();
    nextImage.decoding = 'async';
    nextImage.src = nextCoverImage;
  }, [coverImageIndex, isCoverVisible]);

  if ((!props.rootPaths || props.rootPaths.length === 0) && props.imageList.length === 0) {
    if (!props.appSettings) {
      return null;
    }
    const hasLastPath = !!props.appSettings.lastRootPath || !!props.appSettings.rootFolders?.length;
    const coverImage = COVER_IMAGES[coverImageIndex] ?? COVER_IMAGES[0];

    return (
      <div className="flex-1 flex h-full p-2 bg-transparent">
        <div className="flex w-full h-full bg-bg-secondary rounded-lg border border-border-color/25 overflow-hidden">
          <div className="w-1/2 hidden md:block relative overflow-hidden bg-black">
            <AnimatePresence initial={false}>
              {coverImage && (
                <motion.img
                  alt=""
                  aria-hidden="true"
                  className="absolute -inset-[5%] w-[110%] h-[110%] object-cover blur-2xl opacity-50"
                  key={`${coverImage}-backdrop`}
                  src={coverImage}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 0.5 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: prefersReducedMotion ? 0 : 1.1, ease: 'easeInOut' }}
                />
              )}
            </AnimatePresence>
            <div className="absolute inset-0 bg-black/20" aria-hidden="true" />
            <AnimatePresence initial={false}>
              {coverImage && (
                <motion.img
                  alt=""
                  aria-hidden="true"
                  className="absolute inset-0 w-full h-full object-contain"
                  key={coverImage}
                  src={coverImage}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: prefersReducedMotion ? 0 : 1.1, ease: 'easeInOut' }}
                />
              )}
            </AnimatePresence>
          </div>

          <div className="w-full md:w-1/2 relative overflow-hidden isolate">
            <div className="absolute inset-0 -z-10 pointer-events-none">
              <AnimatePresence initial={false}>
                {coverImage && (
                  <motion.img
                    key={`${coverImage}-ambient`}
                    src={coverImage}
                    className="absolute inset-0 w-full h-full object-cover blur-2xl opacity-50 pointer-events-none"
                    aria-hidden="true"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 0.5 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: prefersReducedMotion ? 0 : 1.1, ease: 'easeInOut' }}
                  />
                )}
              </AnimatePresence>
              <div className="absolute inset-0 bg-bg-secondary/90"></div>
            </div>

            <div className="w-full h-full flex flex-col p-8 lg:p-16 overflow-y-auto custom-scrollbar relative z-10">
              {isSettingsOpen && props.appSettings ? (
                <SettingsPanel
                  appSettings={props.appSettings}
                  onBack={() => setUI({ isSettingsOpen: false })}
                  onLibraryRefresh={props.onLibraryRefresh}
                  onSettingsChange={props.onSettingsChange}
                  rootPaths={props.rootPaths}
                />
              ) : (
                <>
                  <div className="my-auto text-left relative z-10">
                    <Text variant={TextVariants.displayLarge}>{t('library.splash.brand')}</Text>
                    <Text
                      variant={TextVariants.heading}
                      color={TextColors.secondary}
                      weight={TextWeights.normal}
                      className="mb-10 max-w-md drop-shadow-sm"
                    >
                      {hasLastPath ? (
                        <>
                          {t('library.splash.welcomeBack')}
                          <br />
                          {t('library.splash.welcomeBackDesc')}
                        </>
                      ) : props.isAndroid ? (
                        t('library.splash.descriptionAndroid')
                      ) : (
                        t('library.splash.descriptionDesktop')
                      )}
                    </Text>
                    <div className="flex flex-col w-full max-w-xs gap-4 relative z-10">
                      {hasLastPath && (
                        <Button
                          className="rounded-md h-11 w-full flex justify-center items-center shadow-md transition-transform duration-200 hover:scale-[1.01] active:scale-[.98]"
                          onClick={props.onContinueSession}
                          size="lg"
                        >
                          <RefreshCw size={20} className="mr-2" /> {t('library.splash.continueSession')}
                        </Button>
                      )}
                      <div className="flex items-center gap-2">
                        <Button
                          className={`rounded-md grow flex justify-center items-center shadow-md h-11 transition-transform duration-200 hover:scale-[1.01] active:scale-[.98] ${
                            hasLastPath ? 'bg-surface text-text-primary' : ''
                          }`}
                          onClick={props.onOpenFolder}
                          size="lg"
                        >
                          <Folder size={20} className="mr-2" />
                          {props.isAndroid
                            ? t('library.splash.openLibrary')
                            : hasLastPath
                              ? t('library.splash.addFolder')
                              : t('library.splash.openFolder')}
                        </Button>
                        {!props.isAndroid && (
                          <Button
                            className="rounded-md grow flex justify-center items-center bg-surface text-text-primary shadow-md h-11 transition-transform duration-200 hover:scale-[1.01] active:scale-[.98]"
                            onClick={props.onOpenImage}
                            size="lg"
                          >
                            <ImagePlus aria-hidden="true" size={20} className="mr-2" />
                            {t('library.splash.openImage')}
                          </Button>
                        )}
                        <Button
                          aria-label={t('settings.general.title')}
                          className="px-3 bg-surface text-text-primary shadow-md h-11 transition-transform duration-200 hover:scale-[1.03] active:scale-[.96]"
                          onClick={() => setUI({ isSettingsOpen: true })}
                          size="lg"
                          data-tooltip={t('settings.general.title')}
                          type="button"
                          variant="ghost"
                        >
                          <Settings aria-hidden="true" size={20} />
                        </Button>
                      </div>
                    </div>
                  </div>

                  <Text
                    variant={TextVariants.small}
                    as="div"
                    className="absolute bottom-8 left-8 lg:left-16 space-y-1 z-10 drop-shadow-sm"
                  >
                    {appVersion && (
                      <div className="flex items-center space-x-2">
                        <p>
                          <span className="rounded-md py-1">
                            {t('library.splash.version', { version: appVersion })}
                          </span>
                        </p>
                      </div>
                    )}
                  </Text>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full min-w-0 bg-bg-secondary rounded-lg overflow-hidden">
      <header
        className="p-3 shrink-0 flex justify-between items-center border-b border-surface gap-4"
        onMouseEnter={() => setIsProgressHovered(true)}
        onMouseLeave={() => setIsProgressHovered(false)}
      >
        <div className="min-w-0">
          <Text variant={TextVariants.headline}>{t('library.header.title')}</Text>
          {!props.isAndroid && (
            <div className="flex items-center gap-2">
              {props.currentFolderPath ? (
                <Text className="truncate">{props.currentFolderPath}</Text>
              ) : props.imageList.length > 0 ? (
                <Text className="truncate">{t('library.header.selectedFiles', { total: props.imageList.length })}</Text>
              ) : (
                <p className="text-sm invisible select-none pointer-events-none h-5 overflow-hidden"></p>
              )}
              <div
                className={`flex items-center gap-2 overflow-hidden transition-[max-width,opacity] duration-300 whitespace-nowrap ${
                  isBusyDelayed ? 'max-w-xs opacity-100' : 'max-w-0 opacity-0'
                }`}
                onTransitionEnd={(e) => {
                  if (e.propertyName === 'opacity' && !isBusyDelayed) {
                    setIsBusyLoaderMounted(false);
                  }
                }}
              >
                {isBusyLoaderMounted && <Loader2 size={14} className="animate-spin text-text-secondary shrink-0" />}
                <div
                  className={`flex items-center transition-[max-width,opacity] duration-300 ease-out overflow-hidden ${
                    isProgressHovered && isBusyDelayed && (props.thumbnailProgress?.total ?? 0) > 0
                      ? 'max-w-xs opacity-100'
                      : 'max-w-0 opacity-0'
                  }`}
                >
                  <Text variant={TextVariants.small} color={TextColors.secondary} className="whitespace-nowrap">
                    ({props.thumbnailProgress?.current ?? 0}/{props.thumbnailProgress?.total ?? 0})
                  </Text>
                </div>
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 shrink-0">
          {props.importState.status === Status.Importing && (
            <Text as="div" color={TextColors.accent} className="flex items-center gap-2 animate-pulse">
              <FolderInput size={16} />
              <span>
                {t('library.import.progress', {
                  current: props.importState.progress?.current,
                  total: props.importState.progress?.total,
                })}
              </span>
            </Text>
          )}
          {props.importState.status === Status.Success && (
            <Text as="div" color={TextColors.success} className="flex items-center gap-2">
              <Check size={16} />
              <span>{t('library.import.complete')}</span>
            </Text>
          )}
          {props.importState.status === Status.Error && (
            <Text as="div" color={TextColors.error} className="flex items-center gap-2">
              <AlertTriangle size={16} />
              <span>{t('library.import.failed')}</span>
            </Text>
          )}
          <DisplayModeSwitch displayMode={libraryDisplayMode} setDisplayMode={setLibraryDisplayMode} t={t} />

          <div className="flex items-center bg-surface p-1 rounded-lg gap-1 border border-border-color/20">
            <SearchInput indexingProgress={props.indexingProgress} isIndexing={props.isIndexing} />
            <ViewOptionsDropdown
              libraryViewMode={props.libraryViewMode}
              onSelectSize={props.onThumbnailSizeChange}
              onSelectAspectRatio={props.onThumbnailAspectRatioChange}
              onLibraryRefresh={props.onLibraryRefresh}
              setLibraryViewMode={props.setLibraryViewMode}
              thumbnailSize={props.thumbnailSize}
              thumbnailAspectRatio={props.thumbnailAspectRatio}
              thumbnailSizeOptions={translatedThumbnailSizeOptions}
              thumbnailAspectRatioOptions={translatedThumbnailAspectRatioOptions}
              ratingFilterOptions={translatedRatingFilterOptions}
              rawStatusOptions={translatedRawStatusOptions}
              editedStatusOptions={translatedEditedStatusOptions}
              sortOptions={translatedSortOptions}
            />
            {!props.isAndroid && (
              <Button
                aria-label={t('library.splash.openImage')}
                className="h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center"
                data-tooltip={t('library.splash.openImage')}
                onClick={props.onOpenImage}
                type="button"
              >
                <ImagePlus aria-hidden="true" className="w-5 h-5" />
              </Button>
            )}
            <Button
              aria-label={t('library.tooltips.goHome')}
              className="h-12 w-12 bg-transparent text-text-primary shadow-none p-0 flex items-center justify-center"
              onClick={props.onGoHome}
              data-tooltip={t('library.tooltips.goHome')}
              type="button"
            >
              <Home aria-hidden="true" className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </header>

      {props.imageList.length > 0 ? (
        libraryDisplayMode === LibraryDisplayMode.Cull ? (
          <CullingView {...props} />
        ) : (
          <LibraryGrid
            {...props}
            libraryDisplayMode={libraryDisplayMode}
            thumbnailSizeOptions={translatedThumbnailSizeOptions}
          />
        )
      ) : props.isIndexing || props.aiModelDownloadStatus || props.importState.status === Status.Importing ? (
        <div className="flex-1 flex flex-col items-center justify-center" onContextMenu={props.onEmptyAreaContextMenu}>
          <Loader2 className="h-12 w-12 text-secondary animate-spin mb-4" />
          <Text variant={TextVariants.heading} color={TextColors.secondary}>
            {props.aiModelDownloadStatus
              ? t('library.status.downloading', { status: props.aiModelDownloadStatus })
              : props.isIndexing && props.indexingProgress.total > 0
                ? t('library.status.indexing', {
                    current: props.indexingProgress.current,
                    total: props.indexingProgress.total,
                  })
                : props.importState.status === Status.Importing &&
                    props.importState?.progress?.total &&
                    props.importState.progress.total > 0
                  ? t('library.status.importing', {
                      current: props.importState.progress?.current,
                      total: props.importState.progress?.total,
                    })
                  : t('library.status.processing')}
          </Text>
          <Text className="mt-2">{t('library.status.moment')}</Text>
        </div>
      ) : searchCriteria.tags.length > 0 || searchCriteria.text ? (
        <div
          className="flex-1 flex flex-col items-center justify-center text-text-secondary text-center"
          onContextMenu={props.onEmptyAreaContextMenu}
        >
          <Search className="h-12 w-12 text-secondary mb-4" />
          <Text variant={TextVariants.heading} color={TextColors.secondary}>
            {t('library.search.noResults')}
          </Text>
          <Text className="mt-2 max-w-sm">
            {t('library.search.noResultsDesc')}
            {!props.appSettings?.enableAiTagging && t('library.search.noResultsAiHint')}
          </Text>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center" onContextMenu={props.onEmptyAreaContextMenu}>
          <SlidersHorizontal className="h-12 w-12 mb-4 text-text-secondary" />
          <Text>{t('library.filters.noMatch')}</Text>
        </div>
      )}
      {props.isAndroid && (
        <Button
          aria-label={t('library.tooltips.importImages')}
          className="absolute bottom-18 right-8 h-12 w-12 bg-accent text-button-text shadow-lg p-0 flex items-center justify-center z-50 border border-border-color/50"
          onClick={(e) => {
            e.stopPropagation();
            props.onImportClick();
          }}
          data-tooltip={t('library.tooltips.importImages')}
          type="button"
        >
          <FolderInput aria-hidden="true" className="w-6 h-6" />
        </Button>
      )}
    </div>
  );
}
