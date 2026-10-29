import { useCallback, useEffect, useRef, useState } from 'react';
import { GripVertical, Loader2, Maximize2, RotateCcw, RotateCw, ScanLine, Undo2, ZoomIn, ZoomOut } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../../ui/Button';
import { comparePositionFromClientX, comparePositionFromKey } from '../../../utils/compareSlider';
import type { BatchGeometryPreviewItem } from './types';
import { displayFileName } from './utils';

interface BatchGeometryComparisonProps {
  activeItem: BatchGeometryPreviewItem | null;
  activePath: string | null;
  applyDistortion: boolean;
  isProcessing: boolean;
  isRefreshing: boolean;
  items: BatchGeometryPreviewItem[];
  onChoose(path: string): void;
  onRestore(): void;
  onRotate(delta: number): void;
  onToggleGuides(): void;
  orientationLabel(steps: number): string;
  showGuides: boolean;
}

type BatchGeometryViewMode = 'wipe' | 'side-by-side';

export function BatchGeometryComparison({
  activeItem,
  activePath,
  applyDistortion,
  isProcessing,
  isRefreshing,
  items,
  onChoose,
  onRestore,
  onRotate,
  onToggleGuides,
  orientationLabel,
  showGuides,
}: BatchGeometryComparisonProps) {
  const { t } = useTranslation();
  const confidence = activeItem ? Math.round(activeItem.contentConfidence * 100) : 0;
  const [viewMode, setViewMode] = useState<BatchGeometryViewMode>('wipe');
  const [comparePosition, setComparePosition] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const stageRef = useRef<HTMLDivElement>(null);
  const compareDragRef = useRef(false);
  const panStartRef = useRef<{ clientX: number; clientY: number; x: number; y: number } | null>(null);

  useEffect(() => {
    setComparePosition(0.5);
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setIsPanning(false);
  }, [activePath]);

  const setCompareFromClientX = useCallback((clientX: number) => {
    const bounds = stageRef.current?.getBoundingClientRect();
    if (!bounds) return;
    const nextPosition = comparePositionFromClientX(clientX, bounds.left, bounds.width);
    if (nextPosition !== null) setComparePosition(nextPosition);
  }, []);

  const handleComparePointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      const bounds = stageRef.current?.getBoundingClientRect();
      if (!bounds || bounds.width <= 0) return;
      event.preventDefault();
      event.stopPropagation();
      compareDragRef.current = true;
      event.currentTarget.setPointerCapture(event.pointerId);
      setCompareFromClientX(event.clientX);
    },
    [setCompareFromClientX],
  );

  const handleComparePointerMove = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (!compareDragRef.current) return;
      event.preventDefault();
      event.stopPropagation();
      setCompareFromClientX(event.clientX);
    },
    [setCompareFromClientX],
  );

  const handleComparePointerEnd = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!compareDragRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    compareDragRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const handleCompareKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const nextPosition = comparePositionFromKey(comparePosition, event.key, event.shiftKey);
      if (nextPosition === null) return;
      event.preventDefault();
      setComparePosition(nextPosition);
    },
    [comparePosition],
  );

  const handlePreviewWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      event.preventDefault();
      const nextZoom = Math.min(4, Math.max(1, zoom + (event.deltaY < 0 ? 0.25 : -0.25)));
      setZoom(nextZoom);
      if (nextZoom === 1) setPan({ x: 0, y: 0 });
    },
    [zoom],
  );

  const handlePanPointerDown = useCallback(
    (event: React.PointerEvent<HTMLDivElement>) => {
      if (zoom <= 1 || (event.pointerType === 'mouse' && event.button !== 0)) return;
      event.preventDefault();
      panStartRef.current = { clientX: event.clientX, clientY: event.clientY, x: pan.x, y: pan.y };
      setIsPanning(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    },
    [pan.x, pan.y, zoom],
  );

  const handlePanPointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const start = panStartRef.current;
    if (!start) return;
    event.preventDefault();
    setPan({ x: start.x + event.clientX - start.clientX, y: start.y + event.clientY - start.clientY });
  }, []);

  const handlePanPointerEnd = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!panStartRef.current) return;
    panStartRef.current = null;
    setIsPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const resetZoom = useCallback(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const imageTransform = {
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
  };
  const beforeClip = `inset(0 ${Math.round((1 - comparePosition) * 100)}% 0 0)`;
  const directionLabel = orientationLabel(activeItem?.orientationSteps ?? 0);
  const suggestionLabel = orientationLabel(activeItem?.suggestedOrientationSteps ?? 0);
  const hasManualOverride = Boolean(activeItem && activeItem.orientationSteps !== activeItem.suggestedOrientationSteps);

  return (
    <section aria-labelledby="batch-geometry-review-title" className="batch-geometry-review">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-text-primary" id="batch-geometry-review-title">
            {t('modals.batchGeometry.reviewTitle')}
          </h3>
          <p className="mt-1 max-w-2xl text-xs leading-5 text-text-secondary">
            {t('modals.batchGeometry.reviewDescription')}
          </p>
        </div>
        <span className="rounded-full bg-accent/12 px-2.5 py-1 text-xs font-medium text-accent">
          {t('modals.batchGeometry.reviewedCount', { total: items.length })}
        </span>
      </div>

      <div
        aria-label={t('modals.batchGeometry.reviewedCount', { total: items.length })}
        className="batch-geometry-preview-strip mt-3"
      >
        {items.map((item) => {
          const isActive = item.sourcePath === activePath;
          return (
            <button
              aria-pressed={isActive}
              className={`batch-geometry-preview-chip ${isActive ? 'batch-geometry-preview-chip--active' : ''}`}
              disabled={isProcessing}
              key={item.sourcePath}
              onClick={() => onChoose(item.sourcePath)}
              title={item.sourcePath}
              type="button"
            >
              <img alt="" aria-hidden="true" loading="lazy" src={item.afterPreview} />
              <span className="min-w-0 truncate">{displayFileName(item.sourcePath)}</span>
            </button>
          );
        })}
      </div>

      {activeItem ? (
        <div className="mt-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="min-w-0 truncate text-xs font-medium text-text-primary" title={activeItem.sourcePath}>
              {displayFileName(activeItem.sourcePath)}
            </p>
            <div aria-live="polite" className="batch-geometry-preview-status">
              <span
                className="batch-geometry-preview-status__item"
                data-tone={activeItem.autoCorrected ? 'success' : 'neutral'}
              >
                <span>{t('modals.batchGeometry.direction')}</span>
                <strong>{directionLabel}</strong>
              </span>
              <span
                className="batch-geometry-preview-status__item"
                data-tone={!applyDistortion ? 'neutral' : activeItem.distortionApplied ? 'success' : 'warning'}
              >
                <span>{t('modals.batchGeometry.distortion')}</span>
                <strong>
                  {!applyDistortion
                    ? t('modals.batchGeometry.profileDisabled')
                    : activeItem.distortionApplied
                      ? t('modals.batchGeometry.profileApplied')
                      : t('modals.batchGeometry.profileUnavailable')}
                </strong>
              </span>
            </div>
          </div>

          <div className="batch-geometry-preview-stage-wrap mt-3">
            <div
              aria-label={t('modals.batchGeometry.previewStageLabel')}
              className={`batch-geometry-preview-stage ${zoom > 1 ? 'is-zoomed' : ''} ${isPanning ? 'is-panning' : ''}`}
              onPointerCancel={handlePanPointerEnd}
              onPointerDown={handlePanPointerDown}
              onPointerMove={handlePanPointerMove}
              onPointerUp={handlePanPointerEnd}
              onWheel={handlePreviewWheel}
              ref={stageRef}
              role="group"
            >
              {viewMode === 'wipe' ? (
                <div className="batch-geometry-wipe-canvas" style={imageTransform}>
                  <img
                    alt={`${t('modals.batchGeometry.afterPreview')}：${displayFileName(activeItem.sourcePath)}`}
                    className="batch-geometry-preview-image batch-geometry-preview-image--after"
                    draggable={false}
                    src={activeItem.afterPreview}
                  />
                  <div className="batch-geometry-before-layer" style={{ clipPath: beforeClip }}>
                    <img
                      alt=""
                      aria-hidden="true"
                      className="batch-geometry-preview-image"
                      draggable={false}
                      src={activeItem.beforePreview}
                    />
                  </div>
                </div>
              ) : (
                <div className="batch-geometry-side-by-side" style={imageTransform}>
                  <figure className="batch-geometry-preview-frame">
                    <figcaption>{t('modals.batchGeometry.beforePreview')}</figcaption>
                    <img
                      alt={`${t('modals.batchGeometry.beforePreview')}：${displayFileName(activeItem.sourcePath)}`}
                      src={activeItem.beforePreview}
                    />
                  </figure>
                  <figure className="batch-geometry-preview-frame batch-geometry-preview-frame--after">
                    <figcaption>{t('modals.batchGeometry.afterPreview')}</figcaption>
                    <img
                      alt={`${t('modals.batchGeometry.afterPreview')}：${displayFileName(activeItem.sourcePath)}`}
                      src={activeItem.afterPreview}
                    />
                  </figure>
                </div>
              )}

              {viewMode === 'wipe' && (
                <>
                  <span className="batch-geometry-stage-label batch-geometry-stage-label--left">
                    {t('modals.batchGeometry.beforePreview')}
                  </span>
                  <span className="batch-geometry-stage-label batch-geometry-stage-label--right">
                    {t('modals.batchGeometry.afterPreview')}
                  </span>
                  <div
                    aria-label={t('modals.batchGeometry.compareDivider')}
                    aria-orientation="horizontal"
                    aria-valuemax={100}
                    aria-valuemin={0}
                    aria-valuenow={Math.round(comparePosition * 100)}
                    className="batch-geometry-compare-line"
                    onKeyDown={handleCompareKeyDown}
                    onPointerCancel={handleComparePointerEnd}
                    onPointerDown={handleComparePointerDown}
                    onPointerMove={handleComparePointerMove}
                    onPointerUp={handleComparePointerEnd}
                    role="slider"
                    style={{ left: `${comparePosition * 100}%` }}
                    tabIndex={0}
                  >
                    <span>
                      <GripVertical aria-hidden="true" size={17} />
                    </span>
                  </div>
                </>
              )}

              {isRefreshing && (
                <div aria-live="polite" className="batch-geometry-preview-loading">
                  <Loader2 aria-hidden="true" className="animate-spin" size={18} />
                  {t('modals.batchGeometry.previewing')}
                </div>
              )}
            </div>

            <div className="batch-geometry-preview-toolbar">
              <div aria-label={t('modals.batchGeometry.viewMode')} className="batch-geometry-view-mode" role="group">
                <button
                  aria-pressed={viewMode === 'wipe'}
                  className={`batch-geometry-view-mode__button ${viewMode === 'wipe' ? 'is-active' : ''}`}
                  onClick={() => setViewMode('wipe')}
                  type="button"
                >
                  {t('modals.batchGeometry.wipeView')}
                </button>
                <button
                  aria-pressed={viewMode === 'side-by-side'}
                  className={`batch-geometry-view-mode__button ${viewMode === 'side-by-side' ? 'is-active' : ''}`}
                  onClick={() => setViewMode('side-by-side')}
                  type="button"
                >
                  {t('modals.batchGeometry.sideBySideView')}
                </button>
              </div>

              <div className="batch-geometry-preview-zoom" role="group">
                <button
                  aria-label={t('modals.batchGeometry.zoomOut')}
                  className="ui-icon-button ui-icon-button--sm"
                  disabled={zoom <= 1}
                  onClick={() => {
                    const nextZoom = Math.max(1, zoom - 0.25);
                    setZoom(nextZoom);
                    if (nextZoom === 1) setPan({ x: 0, y: 0 });
                  }}
                  title={t('modals.batchGeometry.zoomOut')}
                  type="button"
                >
                  <ZoomOut aria-hidden="true" size={15} />
                </button>
                <span aria-live="polite" className="batch-geometry-preview-zoom__value">
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  aria-label={t('modals.batchGeometry.zoomIn')}
                  className="ui-icon-button ui-icon-button--sm"
                  disabled={zoom >= 4}
                  onClick={() => setZoom((currentZoom) => Math.min(4, currentZoom + 0.25))}
                  title={t('modals.batchGeometry.zoomIn')}
                  type="button"
                >
                  <ZoomIn aria-hidden="true" size={15} />
                </button>
                <button
                  aria-label={t('modals.batchGeometry.resetZoom')}
                  className="ui-icon-button ui-icon-button--sm"
                  disabled={zoom === 1 && pan.x === 0 && pan.y === 0}
                  onClick={resetZoom}
                  title={t('modals.batchGeometry.resetZoom')}
                  type="button"
                >
                  <Maximize2 aria-hidden="true" size={15} />
                </button>
              </div>

              <button
                aria-pressed={showGuides}
                className={`batch-geometry-guide-button ${showGuides ? 'is-active' : ''}`}
                disabled={isProcessing}
                onClick={onToggleGuides}
                type="button"
              >
                <ScanLine aria-hidden="true" size={15} />
                {showGuides ? t('modals.batchGeometry.hideGrid') : t('modals.batchGeometry.showGrid')}
              </button>
            </div>

            {viewMode === 'wipe' && (
              <label className="batch-geometry-compare-control">
                <span>{t('modals.batchGeometry.beforePreview')}</span>
                <input
                  aria-label={t('modals.batchGeometry.compareDivider')}
                  max={100}
                  min={0}
                  onChange={(event) => setComparePosition(Number(event.target.value) / 100)}
                  type="range"
                  value={Math.round(comparePosition * 100)}
                />
                <span>{t('modals.batchGeometry.afterPreview')}</span>
              </label>
            )}
          </div>

          <div aria-live="polite" className="batch-geometry-preview-facts">
            <div>
              <span>{t('modals.batchGeometry.suggestion')}</span>
              <strong>{suggestionLabel}</strong>
              <small>
                {activeItem.autoCorrected
                  ? t('modals.batchGeometry.orientationSuggestion', { confidence, orientation: suggestionLabel })
                  : t('modals.batchGeometry.noOrientationChange')}
              </small>
            </div>
            <div>
              <span>{t('modals.batchGeometry.currentDirection')}</span>
              <strong>{directionLabel}</strong>
              <small>
                {hasManualOverride
                  ? t('modals.batchGeometry.manualOverride')
                  : t('modals.batchGeometry.usingSuggestion')}
              </small>
            </div>
            <div>
              <span>{t('modals.batchGeometry.profileStatus')}</span>
              <strong>
                {!applyDistortion
                  ? t('modals.batchGeometry.profileDisabled')
                  : activeItem.distortionApplied
                    ? t('modals.batchGeometry.profileApplied')
                    : t('modals.batchGeometry.profileUnavailable')}
              </strong>
              <small>{t('modals.batchGeometry.profileStatusHint')}</small>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border-color bg-bg-primary/45 p-2.5">
            <p className="text-xs text-text-secondary">{t('modals.batchGeometry.manualControls')}</p>
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                aria-label={t('modals.batchGeometry.rotateLeft')}
                className="ui-icon-button ui-icon-button--md"
                disabled={isProcessing || isRefreshing}
                onClick={() => onRotate(-1)}
                title={t('modals.batchGeometry.rotateLeft')}
                type="button"
              >
                <RotateCcw aria-hidden="true" size={16} />
              </button>
              <button
                className="batch-geometry-rotation-button"
                disabled={isProcessing || isRefreshing}
                onClick={() => onRotate(2)}
                type="button"
              >
                {t('modals.batchGeometry.rotate180')}
              </button>
              <button
                aria-label={t('modals.batchGeometry.rotateRight')}
                className="ui-icon-button ui-icon-button--md"
                disabled={isProcessing || isRefreshing}
                onClick={() => onRotate(1)}
                title={t('modals.batchGeometry.rotateRight')}
                type="button"
              >
                <RotateCw aria-hidden="true" size={16} />
              </button>
              <Button
                disabled={
                  isProcessing || isRefreshing || activeItem.orientationSteps === activeItem.suggestedOrientationSteps
                }
                onClick={onRestore}
                size="sm"
                type="button"
                variant="ghost"
              >
                <Undo2 aria-hidden="true" size={14} />
                {t('modals.batchGeometry.restoreSuggestion')}
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <div aria-live="polite" className="batch-geometry-preview-stage batch-geometry-preview-stage--loading mt-4">
          <div className="batch-geometry-preview-skeleton" />
          {isProcessing && (
            <div className="batch-geometry-preview-loading">
              <Loader2 aria-hidden="true" className="animate-spin" size={18} />
              {t('modals.batchGeometry.previewing')}
            </div>
          )}
        </div>
      )}

      <p className="mt-3 text-[11px] leading-5 text-text-tertiary">{t('modals.batchGeometry.previewHint')}</p>
    </section>
  );
}
