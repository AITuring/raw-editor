import { useState, useEffect, useMemo, type ReactNode } from 'react';
import { getVersion } from '@tauri-apps/api/app';
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  Folder,
  FolderInput,
  ImagePlus,
  Loader2,
  RefreshCw,
  Settings,
  Search,
  LayoutGrid,
  Layers3,
  Columns,
  SlidersHorizontal,
  Rows3,
  ArrowDownAZ,
  ArrowUpAZ,
  ChevronDown,
  ChevronUp,
  FileQuestion,
  FolderX,
} from 'lucide-react';
import CullingView from './library/CullingView';
import { motion, AnimatePresence, useReducedMotion } from 'framer-motion';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import Dropdown from '../ui/Dropdown';
import TaskProgress from '../ui/TaskProgress';
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
  SortDirection,
} from '../ui/AppProperties';
import { GroupBadgeInfo, GroupId } from '../../utils/imageGrouping';
import { ImportState, Status } from '../ui/ExportImportProperties';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { LibraryContentState, useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';
import SettingsPanel from './SettingsPanel';
import { COVER_IMAGES, COVER_ROTATION_INTERVAL_MS } from '../../config/coverImages';

import LibraryGrid from './library/LibraryGrid';
import { SearchInput, ViewOptionsDropdown } from './library/LibraryHeader';
import { useShallow } from 'zustand/react/shallow';

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
  contentState: LibraryContentState;
  libraryViewMode: LibraryViewMode;
  multiSelectedPaths: Array<string>;
  onClearSelection(): void;
  onContextMenu(event: any, path: string): void;
  onEmptyAreaContextMenu(event: any): void;
  onGoHome(): void;
  onImageClick(path: string, event: any): void;
  onImageDoubleClick(path: string): void;
  onImportClick(): void;
  onLibraryRefresh(): void | Promise<void>;
  onRate(rating: number): void;
  onOpenImage(): void;
  onOpenMultiImageWorkflow(): void;
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
  totalImageCount: number;
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
    <div className="ui-segmented-control">
      <div className="ui-segmented-track">
        <motion.div
          className="ui-segmented-indicator"
          initial={false}
          animate={{
            x: `${safeIndex * 100}%`,
            width: `${100 / options.length}%`,
          }}
          transition={prefersReducedMotion ? { duration: 0 } : { duration: 0.16, ease: [0.22, 1, 0.36, 1] }}
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
              className={clsx('ui-segmented-item', isActive && 'is-active')}
              data-tooltip={opt.tooltip}
              style={{ WebkitTapHighlightColor: 'transparent' }}
              type="button"
            >
              <Icon aria-hidden="true" className="h-4 w-4" />
            </button>
          );
        })}
      </div>
    </div>
  );
}

function LibraryMessageState({
  icon,
  title,
  description,
  actions,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-1 items-center justify-center px-6 py-10 text-center">
      <div className="flex max-w-lg flex-col items-center">
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full border border-border-color bg-surface text-text-secondary">
          {icon}
        </div>
        <h2 className="text-base font-semibold text-text-primary">{title}</h2>
        <p className="mt-1.5 max-w-md text-sm leading-6 text-text-secondary">{description}</p>
        {actions && <div className="mt-5 flex flex-wrap items-center justify-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}

function LibraryLoadingGrid({ label }: { label: string }) {
  return (
    <div aria-busy="true" aria-label={label} className="relative flex-1 overflow-hidden p-3" role="status">
      <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-3 opacity-70">
        {Array.from({ length: 12 }).map((_, index) => (
          <div className="aspect-square overflow-hidden rounded-md bg-surface" key={index}>
            <div className="library-loading-sheen h-full w-full" />
          </div>
        ))}
      </div>
      <div className="absolute inset-0 flex items-center justify-center bg-bg-secondary/35">
        <div className="flex items-center gap-2 rounded-sm border border-border-color bg-bg-secondary px-3 py-2 text-xs text-text-secondary shadow-lg">
          <Loader2 aria-hidden="true" className="animate-spin text-status-info" size={16} />
          {label}
        </div>
      </div>
    </div>
  );
}

export default function MainLibrary(props: MainLibraryProps) {
  const { t } = useTranslation();
  const prefersReducedMotion = useReducedMotion();
  const { isSettingsOpen, libraryContextPanel, setUI } = useUIStore(
    useShallow((state) => ({
      isSettingsOpen: state.isSettingsOpen,
      libraryContextPanel: state.libraryContextPanel,
      setUI: state.setUI,
    })),
  );
  const [coverImageIndex, setCoverImageIndex] = useState(() =>
    COVER_IMAGES.length > 0 ? Math.floor(Math.random() * COVER_IMAGES.length) : 0,
  );
  const [appVersion, setAppVersion] = useState('');
  const [isBusyDelayed, setIsBusyDelayed] = useState(false);
  const [isBusyLoaderMounted, setIsBusyLoaderMounted] = useState(false);
  const [isProgressHovered, setIsProgressHovered] = useState(false);

  const libraryDisplayMode = props.appSettings?.libraryDisplayMode || LibraryDisplayMode.Grid;

  const setLibraryDisplayMode = (mode: LibraryDisplayMode) => {
    if (props.appSettings) {
      props.onSettingsChange({
        ...props.appSettings,
        libraryDisplayMode: mode,
      });
    }
  };

  const handleLibraryViewModeChange = async (mode: LibraryViewMode) => {
    props.setLibraryViewMode(mode);
    if (props.appSettings) {
      await props.onSettingsChange({ ...props.appSettings, libraryViewMode: mode });
    }
    await props.onLibraryRefresh();
  };

  const { filterCriteria, searchCriteria, setFilterCriteria, setSearchCriteria, setSortCriteria, sortCriteria } =
    useLibraryStore(
      useShallow((state) => ({
        filterCriteria: state.filterCriteria,
        searchCriteria: state.searchCriteria,
        setFilterCriteria: state.setFilterCriteria,
        setSearchCriteria: state.setSearchCriteria,
        setSortCriteria: state.setSortCriteria,
        sortCriteria: state.sortCriteria,
      })),
    );

  const locationLabel = useMemo(() => {
    if (!props.currentFolderPath) return t('library.header.selectedFiles', { total: props.totalImageCount });
    if (props.currentFolderPath.startsWith('Album: ')) return props.currentFolderPath.slice('Album: '.length);
    const parts = props.currentFolderPath.split(/[\\/]/).filter(Boolean);
    return parts.at(-1) || props.currentFolderPath;
  }, [props.currentFolderPath, props.totalImageCount, t]);

  const hasSearch = searchCriteria.tags.length > 0 || !!searchCriteria.text;
  const hasFilters =
    filterCriteria.rating !== 0 ||
    (filterCriteria.rawStatus && filterCriteria.rawStatus !== RawStatus.All) ||
    (filterCriteria.editedStatus && filterCriteria.editedStatus !== EditedStatus.All) ||
    (filterCriteria.colors && filterCriteria.colors.length > 0);

  const clearDiscoveryCriteria = () => {
    setSearchCriteria({ tags: [], text: '', mode: 'OR' });
    setFilterCriteria({ colors: [], rating: 0, rawStatus: RawStatus.All, editedStatus: EditedStatus.All });
  };

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

  const isImporting = props.importState.status === Status.Importing;
  const hasThumbnailProgress =
    isBusyDelayed &&
    (props.thumbnailProgress?.total ?? 0) > 0 &&
    (props.thumbnailProgress?.current ?? 0) < (props.thumbnailProgress?.total ?? 0);
  const activeLibraryTask = isImporting
    ? {
        current: props.importState.progress?.current ?? 0,
        indeterminate: (props.importState.progress?.total ?? 0) <= 0,
        label:
          (props.importState.progress?.total ?? 0) > 0
            ? t('library.status.importing', {
                current: props.importState.progress?.current ?? 0,
                total: props.importState.progress?.total ?? 0,
              })
            : t('library.status.processing'),
        total: props.importState.progress?.total ?? 0,
      }
    : props.isIndexing
      ? {
          current: props.indexingProgress.current,
          indeterminate: props.indexingProgress.total <= 0,
          label:
            props.indexingProgress.total > 0
              ? t('library.status.indexing', {
                  current: props.indexingProgress.current,
                  total: props.indexingProgress.total,
                })
              : t('library.status.processing'),
          total: props.indexingProgress.total,
        }
      : props.aiModelDownloadStatus
        ? {
            current: null,
            indeterminate: true,
            label: t('library.status.downloading', { status: props.aiModelDownloadStatus }),
            total: null,
          }
        : hasThumbnailProgress
          ? {
              current: props.thumbnailProgress.current,
              indeterminate: false,
              label: `${t('library.status.processing')} (${props.thumbnailProgress.current}/${props.thumbnailProgress.total})`,
              total: props.thumbnailProgress.total,
            }
          : isBusyDelayed && props.isLoading
            ? {
                current: null,
                indeterminate: true,
                label: t('library.status.processing'),
                total: null,
              }
            : null;

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
    const coverImage = COVER_IMAGES[coverImageIndex] ?? COVER_IMAGES[0];

    return (
      <div className="flex-1 flex h-full p-2 bg-transparent">
        <div className="ui-chrome-panel grid h-full w-full grid-cols-1 md:grid-cols-2">
          <div className="hidden md:block relative min-w-0 overflow-hidden bg-black">
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

          <div className="relative min-w-0 overflow-hidden isolate">
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
                  <div className="w-full max-w-xl mx-auto my-auto text-left relative z-10">
                    <Text variant={TextVariants.displayLarge}>{t('library.splash.brand')}</Text>
                    <Text
                      variant={TextVariants.heading}
                      color={TextColors.secondary}
                      weight={TextWeights.normal}
                      className="mb-10 max-w-md drop-shadow-sm"
                    >
                      {props.isAndroid
                        ? t('library.splash.descriptionAndroid')
                        : t('library.splash.descriptionDesktop')}
                    </Text>
                    <div className="splash-actions-container relative z-10 flex w-full flex-col gap-4">
                      <div className={props.isAndroid ? 'flex items-center gap-2' : 'splash-action-grid'}>
                        <Button
                          className="splash-action-folder flex h-11 min-w-0 justify-center rounded-md transition-transform duration-200 hover:scale-[1.01] active:scale-[.98]"
                          onClick={props.onOpenFolder}
                          size="lg"
                        >
                          <Folder aria-hidden="true" className="shrink-0" size={20} />
                          <span className="truncate">
                            {props.isAndroid ? t('library.splash.openLibrary') : t('library.splash.openFolder')}
                          </span>
                        </Button>
                        {!props.isAndroid && (
                          <Button
                            className="splash-action-image flex h-11 min-w-0 justify-center rounded-md bg-surface text-text-primary transition-transform duration-200 hover:scale-[1.01] active:scale-[.98]"
                            onClick={props.onOpenImage}
                            size="lg"
                          >
                            <ImagePlus aria-hidden="true" className="shrink-0" size={20} />
                            <span className="truncate">{t('library.splash.openImage')}</span>
                          </Button>
                        )}
                        {!props.isAndroid && (
                          <Button
                            className="splash-action-stack flex h-11 min-w-0 justify-center rounded-md bg-surface text-text-primary transition-transform duration-200 hover:scale-[1.01] hover:bg-card-active active:scale-[.98]"
                            data-tooltip={t('library.splash.multiImageSelectionHint')}
                            onClick={props.onOpenMultiImageWorkflow}
                            size="lg"
                            type="button"
                          >
                            <Layers3 aria-hidden="true" className="shrink-0" size={20} />
                            <span className="truncate">{t('modals.imageStack.title')}</span>
                          </Button>
                        )}
                        <Button
                          aria-label={t('settings.general.title')}
                          className="splash-action-settings h-11 w-11 shrink-0 bg-surface px-0 text-text-primary transition-transform duration-200 hover:scale-[1.03] active:scale-[.96]"
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
    <div className="ui-chrome-panel flex h-full min-w-0 flex-1 flex-col">
      <header
        className="ui-toolbar ui-library-toolbar"
        onMouseEnter={() => setIsProgressHovered(true)}
        onMouseLeave={() => setIsProgressHovered(false)}
      >
        <div className="flex min-w-0 items-center gap-2">
          <Button
            aria-label={t('library.tooltips.goHome')}
            className="shrink-0 p-0"
            data-tooltip={t('library.tooltips.goHome')}
            onClick={props.onGoHome}
            size="icon"
            type="button"
            variant="secondary"
          >
            <ArrowLeft aria-hidden="true" className="h-4 w-4" />
          </Button>

          <div className="flex min-w-0 items-center gap-2" title={props.currentFolderPath || locationLabel}>
            <span className="hidden shrink-0 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary lg:inline">
              {t('library.header.title')}
            </span>
            <span aria-hidden="true" className="hidden h-4 w-px shrink-0 bg-border-color lg:block" />
            <span className="truncate text-sm font-semibold text-text-primary">{locationLabel}</span>
            <span className="shrink-0 rounded-sm bg-surface px-1.5 py-0.5 text-[10px] tabular-nums text-text-secondary">
              {props.imageList.length === props.totalImageCount
                ? props.totalImageCount
                : t('library.header.resultCount', {
                    shown: props.imageList.length,
                    total: props.totalImageCount,
                  })}
            </span>
          </div>

          <div
            className={`flex items-center gap-1.5 overflow-hidden whitespace-nowrap transition-[max-width,opacity] duration-200 ${
              isBusyDelayed ? 'max-w-xs opacity-100' : 'max-w-0 opacity-0'
            }`}
            onTransitionEnd={(event) => {
              if (event.propertyName === 'opacity' && !isBusyDelayed) setIsBusyLoaderMounted(false);
            }}
          >
            {isBusyLoaderMounted && (
              <Loader2 aria-hidden="true" className="shrink-0 animate-spin text-status-info" size={13} />
            )}
            <span
              className={`overflow-hidden text-[10px] tabular-nums text-text-secondary transition-[max-width,opacity] duration-200 ${
                isProgressHovered && isBusyDelayed && (props.thumbnailProgress?.total ?? 0) > 0
                  ? 'max-w-28 opacity-100'
                  : 'max-w-0 opacity-0'
              }`}
            >
              {props.thumbnailProgress?.current ?? 0}/{props.thumbnailProgress?.total ?? 0}
            </span>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {props.importState.status !== Status.Idle && (
            <div
              aria-live={props.importState.status === Status.Error ? 'assertive' : 'polite'}
              className="hidden xl:flex"
              role="status"
            >
              {props.importState.status === Status.Importing ? (
                <span className="semantic-status" data-tone="processing">
                  <FolderInput aria-hidden="true" size={14} />
                  <span className="hidden 2xl:inline">
                    {t('library.import.progress', {
                      current: props.importState.progress?.current,
                      total: props.importState.progress?.total,
                    })}
                  </span>
                </span>
              ) : props.importState.status === Status.Success ? (
                <span className="semantic-status" data-tone="success">
                  <Check aria-hidden="true" size={14} />
                  <span className="hidden 2xl:inline">{t('library.import.complete')}</span>
                </span>
              ) : props.importState.status === Status.Error ? (
                <span className="semantic-status" data-tone="error">
                  <AlertTriangle aria-hidden="true" size={14} />
                  <span className="hidden 2xl:inline">{t('library.import.failed')}</span>
                </span>
              ) : null}
            </div>
          )}

          <DisplayModeSwitch displayMode={libraryDisplayMode} setDisplayMode={setLibraryDisplayMode} t={t} />
          <span aria-hidden="true" className="mx-0.5 h-5 w-px bg-border-color" />
          <SearchInput indexingProgress={props.indexingProgress} isIndexing={props.isIndexing} />

          <div className={libraryContextPanel ? 'hidden' : 'hidden items-center gap-1 xl:flex'}>
            {sortCriteria.order === SortDirection.Ascending ? (
              <ArrowDownAZ aria-hidden="true" className="text-text-secondary" size={15} />
            ) : (
              <ArrowUpAZ aria-hidden="true" className="text-text-secondary" size={15} />
            )}
            <Dropdown
              className="w-36"
              onChange={(value) => setSortCriteria((previous) => ({ ...previous, key: value }))}
              options={translatedSortOptions.map((option) => ({ value: option.key, label: option.label }))}
              triggerClassName="mr-0"
              value={sortCriteria.key}
            />
            <button
              aria-label={
                sortCriteria.order === SortDirection.Ascending
                  ? t('library.header.viewOptions.sortDescending')
                  : t('library.header.viewOptions.sortAscending')
              }
              className="ui-icon-button ui-icon-button--md"
              data-tooltip={
                sortCriteria.order === SortDirection.Ascending
                  ? t('library.header.viewOptions.sortDescending')
                  : t('library.header.viewOptions.sortAscending')
              }
              onClick={() =>
                setSortCriteria((previous) => ({
                  ...previous,
                  order:
                    previous.order === SortDirection.Ascending ? SortDirection.Descending : SortDirection.Ascending,
                }))
              }
              type="button"
            >
              {sortCriteria.order === SortDirection.Ascending ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
            </button>
          </div>

          <ViewOptionsDropdown
            libraryViewMode={props.libraryViewMode}
            onSelectSize={props.onThumbnailSizeChange}
            onSelectAspectRatio={props.onThumbnailAspectRatioChange}
            onLibraryRefresh={props.onLibraryRefresh}
            setLibraryViewMode={(mode) => void handleLibraryViewModeChange(mode)}
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
              className="p-0"
              data-tooltip={t('library.splash.openImage')}
              onClick={props.onOpenImage}
              size="icon"
              type="button"
              variant="secondary"
            >
              <ImagePlus aria-hidden="true" className="h-4 w-4" />
            </Button>
          )}
        </div>
      </header>

      {activeLibraryTask && (
        <div className="shrink-0 border-b border-border-color bg-bg-primary/35 px-4 py-2">
          <TaskProgress
            ariaLabel={activeLibraryTask.label}
            compact
            current={activeLibraryTask.current}
            indeterminate={activeLibraryTask.indeterminate}
            label={activeLibraryTask.label}
            total={activeLibraryTask.total}
          />
        </div>
      )}

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
      ) : props.contentState.status === 'loading' || props.importState.status === Status.Importing ? (
        <LibraryLoadingGrid
          label={
            props.importState.status === Status.Importing && (props.importState.progress?.total ?? 0) > 0
              ? t('library.status.importing', {
                  current: props.importState.progress?.current,
                  total: props.importState.progress?.total,
                })
              : t('library.states.loadingTitle')
          }
        />
      ) : props.contentState.status === 'error' ? (
        <LibraryMessageState
          actions={
            <>
              <Button onClick={props.onLibraryRefresh} type="button">
                <RefreshCw aria-hidden="true" size={15} />
                {t('library.actions.retry')}
              </Button>
              <Button className="bg-surface text-text-primary" onClick={props.onOpenFolder} type="button">
                <Folder aria-hidden="true" size={15} />
                {t('library.actions.openAnotherFolder')}
              </Button>
            </>
          }
          description={t('library.states.errorDescription', {
            error: props.contentState.error || t('library.states.unknownError'),
          })}
          icon={<AlertTriangle aria-hidden="true" className="text-status-error" size={22} />}
          title={t('library.states.errorTitle')}
        />
      ) : hasSearch ? (
        <LibraryMessageState
          actions={
            <Button className="bg-surface text-text-primary" onClick={clearDiscoveryCriteria} type="button">
              {t('library.actions.clearSearchAndFilters')}
            </Button>
          }
          description={t('library.search.noResultsDesc')}
          icon={<Search aria-hidden="true" size={21} />}
          title={t('library.search.noResults')}
        />
      ) : hasFilters || props.totalImageCount > 0 ? (
        <LibraryMessageState
          actions={
            <Button className="bg-surface text-text-primary" onClick={clearDiscoveryCriteria} type="button">
              {t('library.actions.clearFilters')}
            </Button>
          }
          description={t('library.states.filteredDescription')}
          icon={<SlidersHorizontal aria-hidden="true" size={21} />}
          title={t('library.filters.noMatch')}
        />
      ) : props.contentState.status === 'unsupported' ? (
        <LibraryMessageState
          actions={
            <>
              <Button className="bg-surface text-text-primary" onClick={props.onOpenFolder} type="button">
                <Folder aria-hidden="true" size={15} />
                {t('library.actions.openAnotherFolder')}
              </Button>
              <Button onClick={props.onImportClick} type="button">
                <FolderInput aria-hidden="true" size={15} />
                {t('library.actions.importHere')}
              </Button>
            </>
          }
          description={t('library.states.unsupportedDescription', { total: props.contentState.totalFiles })}
          icon={<FileQuestion aria-hidden="true" size={22} />}
          title={t('library.states.unsupportedTitle')}
        />
      ) : (
        <LibraryMessageState
          actions={
            <>
              <Button onClick={props.onImportClick} type="button">
                <FolderInput aria-hidden="true" size={15} />
                {t('library.actions.importHere')}
              </Button>
              <Button className="bg-surface text-text-primary" onClick={props.onOpenFolder} type="button">
                <Folder aria-hidden="true" size={15} />
                {t('library.actions.openAnotherFolder')}
              </Button>
            </>
          }
          description={t('library.states.emptyDescription')}
          icon={<FolderX aria-hidden="true" size={22} />}
          title={t('library.states.emptyTitle')}
        />
      )}
      {props.isAndroid && (
        <Button
          aria-label={t('library.tooltips.importImages')}
          className="absolute bottom-18 right-8 z-50 flex h-12 w-12 items-center justify-center bg-accent p-0 text-button-text"
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
