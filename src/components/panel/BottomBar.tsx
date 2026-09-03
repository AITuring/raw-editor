import { useState, useEffect, useId, useLayoutEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import {
  Star,
  Copy,
  ClipboardPaste,
  ChevronUp,
  ChevronDown,
  Check,
  Settings,
  Filter,
  Eye,
  Info,
  Download,
  SquarePen,
  Scan,
} from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';
import { useShallow } from 'zustand/react/shallow';
import { useTranslation } from 'react-i18next';

import Filmstrip from './Filmstrip';
import { FilterCriteria, GLOBAL_KEYS, ImageFile, SelectedImage, ThumbnailAspectRatio } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import { useLibraryStore } from '../../store/useLibraryStore';
import { useUIStore } from '../../store/useUIStore';
import { COLOR_LABELS } from '../../utils/adjustments';

interface BottomBarProps {
  filmstripHeight?: number;
  imageList?: Array<ImageFile>;
  imageRatings?: Record<string, number> | null;
  isCopied: boolean;
  isCopyDisabled: boolean;
  isExportDisabled?: boolean;
  isFilmstripVisible?: boolean;
  isLibraryView?: boolean;
  isLoading?: boolean;
  isPasted: boolean;
  isPasteDisabled: boolean;
  isRatingDisabled?: boolean;
  isResetDisabled?: boolean;
  isResizing?: boolean;
  multiSelectedPaths?: Array<string>;
  onClearSelection?(): void;
  onContextMenu?(event: any, path: string): void;
  onEmptyAreaContextMenu?(event: any): void;
  onCopy(): void;
  onExportClick?(): void;
  onEditSelected?(): void;
  onInfoClick?(): void;
  onImageSelect?(path: string, event: any): void;
  onOpenCopyPasteSettings?(): void;
  onRequestThumbnails?(paths: string[]): void;
  onPaste(): void;
  onQuickPreview?(): void;
  onRate(rate: number): void;
  onReset?(): void;
  onZoomChange?(zoomValue: number, fitToWindow?: boolean): void;
  rating: number;
  selectedImage?: SelectedImage;
  setIsFilmstripVisible?(isVisible: boolean): void;
  showFilmstrip?: boolean;
  showZoomControls?: boolean;
  thumbnailAspectRatio: ThumbnailAspectRatio;
  totalImages?: number;
}

interface StarRatingProps {
  disabled: boolean;
  onRate(rate: number): void;
  rating: number;
}

const StarRating = ({ rating, onRate, disabled }: StarRatingProps) => {
  const { t } = useTranslation();

  return (
    <div
      aria-label={t('library.quickPreview.rating')}
      className={clsx('editor-rating-group', disabled && 'cursor-not-allowed')}
      role="group"
    >
      {[...Array(5)].map((_, index: number) => {
        const starValue = index + 1;
        return (
          <button
            aria-label={
              disabled
                ? t('ui.bottomBar.tooltips.selectToRate')
                : t('ui.bottomBar.tooltips.rateStars', { count: starValue })
            }
            className="editor-rating-button disabled:cursor-not-allowed"
            disabled={disabled}
            key={starValue}
            onClick={() => !disabled && onRate(starValue === rating ? 0 : starValue)}
            data-tooltip={
              disabled
                ? t('ui.bottomBar.tooltips.selectToRate')
                : t('ui.bottomBar.tooltips.rateStars', { count: starValue })
            }
          >
            <Star
              size={15}
              className={clsx(
                'transition-colors duration-150',
                disabled
                  ? 'text-text-secondary opacity-40'
                  : starValue <= rating
                    ? 'fill-accent text-accent'
                    : 'text-text-secondary hover:text-accent',
              )}
            />
          </button>
        );
      })}
    </div>
  );
};

interface QuickFilterProps {
  allColors: Array<{ color: string; name: string }>;
  filterCriteria: FilterCriteria;
  isActive: boolean;
  setFilterCriteria(criteria: Partial<FilterCriteria> | ((previous: FilterCriteria) => FilterCriteria)): void;
}

const QuickFilter = ({ allColors, filterCriteria, isActive, setFilterCriteria }: QuickFilterProps) => {
  const { t } = useTranslation();
  const popoverId = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [position, setPosition] = useState<{ bottom: string; triggerCenter: number } | null>(null);
  const activeColorCount = filterCriteria.colors?.length ?? 0;
  const hasQuickFilters = filterCriteria.rating > 0 || activeColorCount > 0;

  useLayoutEffect(() => {
    if (!isOpen) return;

    const updatePosition = () => {
      const rect = triggerRef.current?.getBoundingClientRect();
      if (!rect) return;

      setPosition({
        bottom: `max(var(--ui-floating-viewport-inset), calc(${window.innerHeight - rect.top}px + var(--ui-filter-popover-trigger-gap)))`,
        triggerCenter: rect.left + rect.width / 2,
      });
    };

    updatePosition();
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);

    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!triggerRef.current?.contains(target) && !popoverRef.current?.contains(target)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
        triggerRef.current?.focus();
      }
    };

    document.addEventListener('pointerdown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('pointerdown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isActive) setIsOpen(false);
  }, [isActive]);

  const clearQuickFilters = () => {
    setFilterCriteria((previous) => ({ ...previous, colors: [], rating: 0 }));
  };

  return (
    <div className="editor-quick-filter shrink-0">
      <button
        aria-controls={popoverId}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        aria-label={t('ui.bottomBar.tooltips.quickFilter')}
        className={clsx('editor-status-icon-button relative', (isOpen || hasQuickFilters) && 'is-active')}
        data-tooltip={t('ui.bottomBar.tooltips.quickFilter')}
        onClick={() => setIsOpen((open) => !open)}
        ref={triggerRef}
        type="button"
      >
        <Filter aria-hidden="true" size={16} />
        {hasQuickFilters && <span aria-hidden="true" className="editor-filter-active-dot" />}
      </button>

      {isOpen &&
        position &&
        typeof document !== 'undefined' &&
        createPortal(
          <div
            aria-label={t('ui.bottomBar.tooltips.quickFilter')}
            aria-modal="false"
            className="editor-filter-popover"
            id={popoverId}
            ref={popoverRef}
            role="dialog"
            style={{
              bottom: position.bottom,
              left: `clamp(var(--ui-floating-viewport-inset), calc(${position.triggerCenter}px - (var(--ui-filter-popover-width) / 2)), calc(100vw - var(--ui-filter-popover-width) - var(--ui-floating-viewport-inset)))`,
            }}
          >
            {hasQuickFilters && (
              <div className="editor-filter-popover-header">
                <span>{t('ui.bottomBar.tooltips.quickFilter')}</span>
                <button className="editor-filter-clear" onClick={clearQuickFilters} type="button">
                  {t('library.actions.clearFilters')}
                </button>
              </div>
            )}

            <div className={clsx('editor-filter-section', !hasQuickFilters && 'is-first')}>
              <span className="editor-filter-section-label">{t('library.header.viewOptions.filterByRating')}</span>
              <div className="editor-filter-options" role="group">
                {[1, 2, 3, 4, 5].map((starValue) => {
                  const isFilled = filterCriteria.rating > 0 && starValue <= filterCriteria.rating;
                  return (
                    <button
                      aria-label={t('ui.bottomBar.tooltips.rateStars', { count: starValue })}
                      aria-pressed={filterCriteria.rating === starValue}
                      className="editor-filter-option-button"
                      key={`qf-star-${starValue}`}
                      onClick={() =>
                        setFilterCriteria((previous) => ({
                          ...previous,
                          rating: previous.rating === starValue ? 0 : starValue,
                        }))
                      }
                      type="button"
                    >
                      <Star
                        aria-hidden="true"
                        className={clsx(
                          'transition-colors duration-150',
                          isFilled ? 'fill-accent text-accent' : 'text-text-secondary',
                        )}
                        size={16}
                      />
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="editor-filter-section">
              <span className="editor-filter-section-label">{t('library.header.viewOptions.filterByColorLabel')}</span>
              <div className="editor-filter-options" role="group">
                {allColors.map((color) => {
                  const isSelected = (filterCriteria.colors || []).includes(color.name);
                  const tooltipTitle =
                    color.name === 'none'
                      ? t('library.header.viewOptions.noLabel')
                      : t(`contextMenus.colors.${color.name}`, {
                          defaultValue: color.name.charAt(0).toUpperCase() + color.name.slice(1),
                        });

                  return (
                    <button
                      aria-label={tooltipTitle}
                      aria-pressed={isSelected}
                      className="editor-color-filter-button"
                      data-tooltip={tooltipTitle}
                      key={`qf-color-${color.name}`}
                      onClick={() => {
                        const currentColors = filterCriteria.colors || [];
                        const colors = currentColors.includes(color.name)
                          ? currentColors.filter((currentColor) => currentColor !== color.name)
                          : [...currentColors, color.name];
                        setFilterCriteria((previous) => ({ ...previous, colors }));
                      }}
                      type="button"
                    >
                      <span
                        className={clsx('editor-color-filter-swatch', isSelected && 'is-selected')}
                        style={{ backgroundColor: color.color }}
                      >
                        {isSelected && <Check aria-hidden="true" className="text-white drop-shadow-md" size={10} />}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
};

export default function BottomBar({
  filmstripHeight,
  imageList = [],
  imageRatings,
  isCopied,
  isCopyDisabled,
  isExportDisabled = false,
  isFilmstripVisible,
  isLibraryView = false,
  isLoading = false,
  isPasted,
  isPasteDisabled,
  isRatingDisabled = false,
  isResizing,
  multiSelectedPaths = [],
  onClearSelection,
  onContextMenu,
  onEmptyAreaContextMenu,
  onCopy,
  onExportClick,
  onEditSelected,
  onInfoClick,
  onImageSelect,
  onOpenCopyPasteSettings,
  onRequestThumbnails,
  onPaste,
  onQuickPreview,
  onRate,
  onZoomChange = () => {},
  rating,
  selectedImage,
  setIsFilmstripVisible,
  showFilmstrip = true,
  showZoomControls = true,
  thumbnailAspectRatio,
  totalImages,
}: BottomBarProps) {
  const { t } = useTranslation();
  const { activeView, isInstantTransition } = useUIStore(
    useShallow((state) => ({
      activeView: state.activeView,
      isInstantTransition: state.isInstantTransition,
    })),
  );

  const { displaySize, originalSize } = useEditorStore(
    useShallow((state) => ({
      displaySize: state.displaySize,
      originalSize: state.originalSize,
    })),
  );

  const [isEditingPercent, setIsEditingPercent] = useState(false);
  const [percentInputValue, setPercentInputValue] = useState('');
  const isDraggingSlider = useRef(false);
  const [isZoomActive, setIsZoomActive] = useState(false);

  const percentInputRef = useRef<HTMLInputElement>(null);
  const isZoomReady = !isLoading && originalSize && originalSize.width > 0 && displaySize && displaySize.width > 0;

  const currentOriginalPercent = isZoomReady
    ? (displaySize.width * (typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1)) / originalSize.width
    : 1.0;

  const [latchedSliderValue, setLatchedSliderValue] = useState(1.0);
  const [latchedDisplayPercent, setLatchedDisplayPercent] = useState(100);

  const numSelected = multiSelectedPaths.length;
  const total = totalImages ?? 0;
  const showSelectionCounter = numSelected > 0;

  const { filterCriteria, setFilterCriteria } = useLibraryStore(
    useShallow((state) => ({
      filterCriteria: state.filterCriteria,
      setFilterCriteria: state.setFilterCriteria,
    })),
  );

  const allColors = [...COLOR_LABELS, { name: 'none', color: '#9ca3af' }];
  const currentHeight = filmstripHeight ?? 120;
  const isCollapsed = !isFilmstripVisible;
  const effectiveHeight = isFilmstripVisible ? currentHeight : 0;
  const shouldAnimate = !isInstantTransition && (!isResizing || isCollapsed);
  const isActiveView = isLibraryView ? activeView === 'library' : activeView === 'editor';

  useEffect(() => {
    if (isZoomReady && !isDraggingSlider.current) {
      setLatchedSliderValue(currentOriginalPercent);
      setLatchedDisplayPercent(Math.round(currentOriginalPercent * 100));
    }
  }, [currentOriginalPercent, isZoomReady]);

  useEffect(() => {
    const handleDragEndGlobal = () => {
      if (isZoomActive) {
        setIsZoomActive(false);
        isDraggingSlider.current = false;
        if (isZoomReady) {
          setLatchedDisplayPercent(Math.round(currentOriginalPercent * 100));
        }
      }
    };

    if (isZoomActive) {
      window.addEventListener('mouseup', handleDragEndGlobal);
      window.addEventListener('touchend', handleDragEndGlobal);
    }

    return () => {
      window.removeEventListener('mouseup', handleDragEndGlobal);
      window.removeEventListener('touchend', handleDragEndGlobal);
    };
  }, [isZoomActive, isZoomReady, currentOriginalPercent]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newZoom = parseFloat(e.target.value);
    setLatchedSliderValue(newZoom);
    setLatchedDisplayPercent(Math.round(newZoom * 100));
    onZoomChange(newZoom);
  };

  const handleMouseDown = () => {
    isDraggingSlider.current = true;
    setIsZoomActive(true);
  };

  const handleMouseUp = () => {
    isDraggingSlider.current = false;
    setIsZoomActive(false);
    if (isZoomReady) {
      setLatchedDisplayPercent(Math.round(currentOriginalPercent * 100));
    }
  };

  const handleZoomKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && ['z', 'y'].includes(e.key.toLowerCase())) {
      (e.target as HTMLElement).blur();
      return;
    }
    if (GLOBAL_KEYS.includes(e.key)) {
      (e.target as HTMLElement).blur();
    }
  };

  const handleResetZoom = () => {
    onZoomChange(0, true);
  };

  const handlePercentClick = () => {
    if (!isZoomReady) return;
    setIsEditingPercent(true);
    setPercentInputValue(latchedDisplayPercent.toString());
    setTimeout(() => {
      percentInputRef.current?.focus();
      percentInputRef.current?.select();
    }, 0);
  };

  const handlePercentSubmit = () => {
    const value = parseFloat(percentInputValue);
    if (!isNaN(value)) {
      const originalPercent = value / 100;
      const clampedPercent = Math.max(0.1, Math.min(2.0, originalPercent));
      onZoomChange(clampedPercent);
    }
    setIsEditingPercent(false);
    setPercentInputValue('');
  };

  const handlePercentKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handlePercentSubmit();
    else if (e.key === 'Escape') {
      setIsEditingPercent(false);
      setPercentInputValue('');
    }
    e.stopPropagation();
  };

  return (
    <div className={clsx('editor-bottom-dock shrink-0', isLibraryView && 'is-library')}>
      {!isLibraryView && showFilmstrip && (
        <div
          className={clsx(
            'overflow-hidden shrink-0 relative',
            shouldAnimate && 'transition-[height] duration-200 ease-out',
          )}
          style={{ height: `${effectiveHeight}px` }}
        >
          <div
            className={clsx(
              'w-full px-1.5 py-1.5 transition-opacity duration-150 ease-out',
              isCollapsed ? 'opacity-0 pointer-events-none' : 'opacity-100 pointer-events-auto',
            )}
            style={{ height: `${currentHeight}px` }}
          >
            <Filmstrip
              imageList={imageList}
              imageRatings={imageRatings}
              isLoading={isLoading}
              multiSelectedPaths={multiSelectedPaths}
              onClearSelection={onClearSelection}
              onContextMenu={onContextMenu}
              onEmptyAreaContextMenu={onEmptyAreaContextMenu}
              onImageSelect={onImageSelect}
              onRequestThumbnails={onRequestThumbnails}
              selectedImage={selectedImage}
              thumbnailAspectRatio={thumbnailAspectRatio}
            />
          </div>
        </div>
      )}

      <div
        className={clsx(
          'editor-status-bar shrink-0',
          isLibraryView ? 'is-library' : 'is-editor',
          !isLibraryView && showFilmstrip && isFilmstripVisible && 'has-filmstrip-divider',
        )}
      >
        <div className="editor-status-primary">
          <StarRating rating={rating} onRate={onRate} disabled={isRatingDisabled} />
          <span aria-hidden="true" className="editor-status-divider is-after-rating" />
          <div className="editor-copy-actions">
            <button
              aria-label={t('ui.bottomBar.tooltips.copySettings')}
              className="editor-status-icon-button relative"
              disabled={isCopyDisabled}
              onClick={onCopy}
              data-tooltip={t('ui.bottomBar.tooltips.copySettings')}
              type="button"
            >
              <AnimatePresence mode="wait" initial={false}>
                {isCopied ? (
                  <motion.div
                    key="copied"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <Check size={16} className="text-status-success" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="copy"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <Copy size={16} />
                  </motion.div>
                )}
              </AnimatePresence>
            </button>

            <button
              aria-label={t('ui.bottomBar.tooltips.pasteSettings')}
              className="editor-status-icon-button relative"
              disabled={isPasteDisabled}
              onClick={onPaste}
              data-tooltip={t('ui.bottomBar.tooltips.pasteSettings')}
              type="button"
            >
              <AnimatePresence mode="wait" initial={false}>
                {isPasted ? (
                  <motion.div
                    key="pasted"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <Check size={16} className="text-status-success" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="paste"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    transition={{ duration: 0.15 }}
                    className="absolute"
                  >
                    <ClipboardPaste size={16} />
                  </motion.div>
                )}
              </AnimatePresence>
            </button>

            <button
              aria-label={t('ui.bottomBar.tooltips.copyPasteSettings')}
              className="editor-status-icon-button"
              onClick={onOpenCopyPasteSettings}
              data-tooltip={t('ui.bottomBar.tooltips.copyPasteSettings')}
              type="button"
            >
              <Settings size={16} />
            </button>
          </div>

          <span aria-hidden="true" className="editor-status-divider is-before-filter" />

          <QuickFilter
            allColors={allColors}
            filterCriteria={filterCriteria}
            isActive={isActiveView}
            setFilterCriteria={setFilterCriteria}
          />

          {showSelectionCounter && (
            <div className="editor-status-selection">
              <span aria-hidden="true" className="editor-status-divider" />
              <span className="editor-status-selection-label">
                {t('ui.bottomBar.imagesSelected', { current: numSelected, total })}
              </span>
            </div>
          )}
        </div>
        {isLibraryView ? (
          <div className="editor-library-status-actions">
            <button
              aria-label={t('library.actions.quickPreview')}
              className="editor-library-action"
              data-tooltip={t('library.actions.quickPreviewHint')}
              disabled={numSelected === 0}
              onClick={onQuickPreview}
              type="button"
            >
              <Eye aria-hidden="true" size={15} />
              <span className="editor-library-action-label">{t('library.actions.quickPreview')}</span>
            </button>
            <button
              aria-label={t('library.actions.showInfo')}
              className="editor-library-action"
              data-tooltip={t('library.actions.showInfo')}
              disabled={numSelected === 0}
              onClick={onInfoClick}
              type="button"
            >
              <Info aria-hidden="true" size={15} />
              <span className="editor-library-action-label">{t('library.actions.info')}</span>
            </button>
            <button
              aria-label={t('ui.bottomBar.tooltips.export')}
              className="editor-library-action"
              data-tooltip={t('ui.bottomBar.tooltips.export')}
              disabled={isExportDisabled}
              onClick={onExportClick}
              type="button"
            >
              <Download aria-hidden="true" size={15} />
              <span className="editor-library-action-label">{t('ui.bottomBar.tooltips.export')}</span>
            </button>
            <span aria-hidden="true" className="mx-1 h-5 w-px bg-surface" />
            <button
              className="editor-footer-button is-primary flex items-center justify-center gap-1.5"
              disabled={numSelected === 0}
              onClick={onEditSelected}
              type="button"
            >
              <SquarePen aria-hidden="true" size={14} />
              {t('library.actions.enterEdit')}
            </button>
          </div>
        ) : (
          <>
            <div className="editor-status-center">
              {showZoomControls && (
                <div className="editor-zoom-control">
                  <button
                    aria-label={t('ui.bottomBar.tooltips.resetZoom')}
                    className="editor-status-icon-button editor-zoom-fit-button"
                    data-tooltip={t('ui.bottomBar.tooltips.resetZoom')}
                    disabled={!isZoomReady}
                    onClick={handleResetZoom}
                    type="button"
                  >
                    <Scan aria-hidden="true" size={15} />
                  </button>

                  <div className="editor-zoom-slider">
                    <div aria-hidden="true" className="editor-zoom-slider-track" />
                    <input
                      aria-label={t('ui.bottomBar.zoomLabel')}
                      className={clsx('slider-input', isZoomActive && 'slider-thumb-active')}
                      disabled={!isZoomReady}
                      max={2.0}
                      min={0.1}
                      onChange={handleSliderChange}
                      onDoubleClick={handleResetZoom}
                      onKeyDown={handleZoomKeyDown}
                      onMouseDown={handleMouseDown}
                      onMouseUp={handleMouseUp}
                      onTouchEnd={handleMouseUp}
                      onTouchStart={handleMouseDown}
                      step="0.05"
                      type="range"
                      value={latchedSliderValue}
                    />
                  </div>

                  <div className="editor-zoom-value">
                    {isEditingPercent ? (
                      <input
                        aria-label={t('ui.bottomBar.tooltips.customZoom')}
                        className="editor-zoom-input"
                        inputMode="numeric"
                        onBlur={handlePercentSubmit}
                        onChange={(event) => setPercentInputValue(event.target.value)}
                        onKeyDown={handlePercentKeyDown}
                        ref={percentInputRef}
                        type="text"
                        value={percentInputValue}
                      />
                    ) : (
                      <button
                        className="editor-zoom-value-button"
                        data-tooltip={t('ui.bottomBar.tooltips.customZoom')}
                        disabled={!isZoomReady}
                        onClick={handlePercentClick}
                        type="button"
                      >
                        {latchedDisplayPercent}%
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="editor-status-trailing">
              {showFilmstrip && (
                <button
                  aria-label={
                    isFilmstripVisible
                      ? t('ui.bottomBar.tooltips.collapseFilmstrip')
                      : t('ui.bottomBar.tooltips.expandFilmstrip')
                  }
                  className="editor-status-icon-button"
                  data-tooltip={
                    isFilmstripVisible
                      ? t('ui.bottomBar.tooltips.collapseFilmstrip')
                      : t('ui.bottomBar.tooltips.expandFilmstrip')
                  }
                  onClick={() => setIsFilmstripVisible?.(!isFilmstripVisible)}
                  type="button"
                >
                  {isFilmstripVisible ? (
                    <ChevronDown aria-hidden="true" size={16} />
                  ) : (
                    <ChevronUp aria-hidden="true" size={16} />
                  )}
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
