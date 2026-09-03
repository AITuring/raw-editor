import { memo, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Eye, EyeOff, Loader2, Maximize, Redo, Undo } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';

import { useLibraryStore } from '../../../store/useLibraryStore';
import { useSettingsStore } from '../../../store/useSettingsStore';
import { findGroupVariants, getVariantLabel } from '../../../utils/imageGrouping';
import { GroupingMode, SelectedImage } from '../../ui/AppProperties';

interface EditorToolbarProps {
  adjustmentsHistory: any[];
  adjustmentsHistoryIndex: number;
  canRedo: boolean;
  canUndo: boolean;
  goToAdjustmentsHistoryIndex(index: number): void;
  isAndroid: boolean;
  isLoading: boolean;
  onBackToLibrary(): void;
  onImageSelect?(path: string, event?: any): void;
  onRedo(): void;
  onToggleFullScreen(): void;
  onToggleShowOriginal(): void;
  onUndo(): void;
  selectedImage: SelectedImage;
  showOriginal: boolean;
}

const EditorToolbar = memo(
  ({
    adjustmentsHistory,
    adjustmentsHistoryIndex,
    canRedo,
    canUndo,
    goToAdjustmentsHistoryIndex,
    isAndroid,
    isLoading,
    onBackToLibrary,
    onImageSelect,
    onRedo,
    onToggleFullScreen,
    onToggleShowOriginal,
    onUndo,
    selectedImage,
    showOriginal,
  }: EditorToolbarProps) => {
    const { t } = useTranslation();
    const [isHistoryVisible, setIsHistoryVisible] = useState(false);
    const historyContainerRef = useRef<HTMLDivElement>(null);

    const imageList = useLibraryStore((state) => state.imageList);
    const groupingMode: GroupingMode = useSettingsStore((state) => state.appSettings?.grouping) ?? 'off';

    const variantOptions = useMemo(() => {
      if (groupingMode === 'off' || !onImageSelect) return [];
      if (selectedImage.path.includes('?vc=')) return [];

      const variants = findGroupVariants(imageList, selectedImage.group_id);
      if (variants.length < 2) return [];
      return variants.map((variant) => ({ path: variant.path, label: getVariantLabel(variant.path) }));
    }, [groupingMode, imageList, onImageSelect, selectedImage.group_id, selectedImage.path]);

    const imageIdentity = useMemo(() => {
      const [sourcePath, virtualCopyId] = selectedImage.path.split('?vc=');
      return {
        baseName: sourcePath.split(/[\\/]/).pop() || '',
        virtualCopyId,
      };
    }, [selectedImage.path]);

    useEffect(() => {
      if (!isHistoryVisible) return;

      const handlePointerDown = (event: MouseEvent) => {
        if (!historyContainerRef.current?.contains(event.target as Node)) {
          setIsHistoryVisible(false);
        }
      };

      document.addEventListener('mousedown', handlePointerDown);
      return () => document.removeEventListener('mousedown', handlePointerDown);
    }, [isHistoryVisible]);

    const toggleHistory = (event: React.MouseEvent) => {
      event.preventDefault();
      if (adjustmentsHistory.length > 1) {
        setIsHistoryVisible((visible) => !visible);
      }
    };

    return (
      <header className="ui-toolbar editor-command-bar">
        <div className="editor-command-group min-w-0">
          <button
            aria-label={t('editor.toolbar.tooltips.backToLibrary')}
            className="editor-command-button"
            data-bench-id="back-to-library"
            data-tooltip={t('editor.toolbar.tooltips.backToLibrary')}
            onClick={onBackToLibrary}
            type="button"
          >
            <ArrowLeft aria-hidden="true" size={17} strokeWidth={1.8} />
          </button>

          <span aria-hidden="true" className="editor-command-divider" />

          <div className="min-w-0 leading-tight">
            <div className="flex min-w-0 items-center gap-1.5">
              <span className="truncate text-[12px] font-medium text-text-primary">{imageIdentity.baseName}</span>
              {imageIdentity.virtualCopyId && (
                <span className="editor-file-badge">
                  {t('editor.toolbar.vc')}-{imageIdentity.virtualCopyId}
                </span>
              )}
              {isLoading && <Loader2 aria-hidden="true" className="animate-spin text-status-info" size={11} />}
            </div>
            {!isAndroid && selectedImage.width > 0 && selectedImage.height > 0 && (
              <span className="block text-[10px] tabular-nums text-text-secondary">
                {selectedImage.width} × {selectedImage.height}
              </span>
            )}
          </div>
        </div>

        {variantOptions.length > 0 && (
          <div className="editor-variant-switcher">
            {variantOptions.map((variant) => {
              const isActive = variant.path === selectedImage.path;
              return (
                <button
                  aria-pressed={isActive}
                  className={clsx('editor-variant-button', isActive && 'is-active')}
                  disabled={isActive}
                  key={variant.path}
                  onClick={(event) => onImageSelect?.(variant.path, event)}
                  type="button"
                >
                  {variant.label}
                </button>
              );
            })}
          </div>
        )}

        <div className="editor-command-group ml-auto" ref={historyContainerRef}>
          <div className="relative flex items-center gap-0.5">
            <button
              aria-label={t('editor.toolbar.tooltips.undo')}
              className="editor-command-button"
              data-bench-id="undo"
              data-tooltip={t('editor.toolbar.tooltips.undo')}
              disabled={!canUndo}
              onClick={onUndo}
              onContextMenu={toggleHistory}
              type="button"
            >
              <Undo aria-hidden="true" size={16} strokeWidth={1.8} />
            </button>
            <button
              aria-label={t('editor.toolbar.tooltips.redo')}
              className="editor-command-button"
              data-tooltip={t('editor.toolbar.tooltips.redo')}
              disabled={!canRedo}
              onClick={onRedo}
              onContextMenu={toggleHistory}
              type="button"
            >
              <Redo aria-hidden="true" size={16} strokeWidth={1.8} />
            </button>

            {isHistoryVisible && (
              <div className="editor-history-popover" role="menu">
                {adjustmentsHistory.map((_, index) => {
                  const isCurrent = index === adjustmentsHistoryIndex;
                  const isFuture = index > adjustmentsHistoryIndex;
                  return (
                    <button
                      aria-current={isCurrent ? 'step' : undefined}
                      className={clsx('editor-history-item', isCurrent && 'is-current', isFuture && 'is-future')}
                      key={index}
                      onClick={() => {
                        goToAdjustmentsHistoryIndex(index);
                        setIsHistoryVisible(false);
                      }}
                      role="menuitem"
                      type="button"
                    >
                      <span>{t('editor.adjustments.title')}</span>
                      <span className="tabular-nums text-text-secondary">{String(index).padStart(2, '0')}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          <span aria-hidden="true" className="editor-command-divider" />

          <button
            aria-label={
              showOriginal ? t('editor.toolbar.tooltips.showEdited') : t('editor.toolbar.tooltips.showOriginal')
            }
            aria-pressed={showOriginal}
            className={clsx('editor-command-button', showOriginal && 'is-active')}
            data-tooltip={
              showOriginal ? t('editor.toolbar.tooltips.showEdited') : t('editor.toolbar.tooltips.showOriginal')
            }
            onClick={onToggleShowOriginal}
            type="button"
          >
            {showOriginal ? (
              <EyeOff aria-hidden="true" size={17} strokeWidth={1.8} />
            ) : (
              <Eye aria-hidden="true" size={17} strokeWidth={1.8} />
            )}
          </button>
          <button
            aria-label={t('editor.toolbar.tooltips.fullscreen')}
            className="editor-command-button"
            data-tooltip={t('editor.toolbar.tooltips.fullscreen')}
            onClick={onToggleFullScreen}
            type="button"
          >
            <Maximize aria-hidden="true" size={16} strokeWidth={1.8} />
          </button>
        </div>
      </header>
    );
  },
);

EditorToolbar.displayName = 'EditorToolbar';

export default EditorToolbar;
