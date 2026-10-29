import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { homeDir } from '@tauri-apps/api/path';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import { CheckCircle2, Eye, FolderOpen, ImagePlus, Loader2, ScanLine, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import Button from '../ui/Button';
import Dropdown from '../ui/Dropdown';
import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import TaskProgress from '../ui/TaskProgress';
import { Invokes } from '../ui/AppProperties';
import { disposeTauriListener } from '../../utils/tauriListenerCleanup';
import type {
  BatchOutputFormat,
  ProcessingPhase,
  BatchGeometryModalProps,
  BatchGeometryProgress,
  BatchGeometryPreviewItem,
  BatchGeometryAnalysisResult,
  BatchGeometryCorrectionResult,
} from './BatchGeometryModal/types';
import {
  normalizeExtensions,
  pathsFromSelection,
  displayFileName,
  getSupportedImageTypes,
  errorMessage,
  ORIENTATION_KEYS,
  normalizeOrientationSteps,
} from './BatchGeometryModal/utils';
import { BatchGeometryComparison } from './BatchGeometryModal/BatchGeometryComparison';

export default function BatchGeometryModal({ isOpen, onClose }: BatchGeometryModalProps) {
  const { t } = useTranslation();
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const backdropPointerTarget = useRef<EventTarget | null>(null);
  const [sourcePaths, setSourcePaths] = useState<string[]>([]);
  const [outputFolder, setOutputFolder] = useState('');
  const [outputFormat, setOutputFormat] = useState<BatchOutputFormat>('jpg');
  const [jpegQuality, setJpegQuality] = useState(92);
  const [applyDistortion, setApplyDistortion] = useState(true);
  const [processingPhase, setProcessingPhase] = useState<ProcessingPhase>('idle');
  const [progress, setProgress] = useState<BatchGeometryProgress | null>(null);
  const [previewItems, setPreviewItems] = useState<BatchGeometryPreviewItem[]>([]);
  const [activePreviewPath, setActivePreviewPath] = useState<string | null>(null);
  const [refreshingPreviewPath, setRefreshingPreviewPath] = useState<string | null>(null);
  const [showGuides, setShowGuides] = useState(false);
  const [previewsStale, setPreviewsStale] = useState(false);
  const [result, setResult] = useState<BatchGeometryCorrectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const previewRefreshRequestIdRef = useRef(0);

  const outputFormatOptions = useMemo<Array<{ label: string; value: BatchOutputFormat }>>(
    () => [
      { label: 'JPEG', value: 'jpg' },
      { label: 'PNG', value: 'png' },
      { label: 'TIFF', value: 'tiff' },
    ],
    [],
  );
  const isProcessing = processingPhase !== 'idle';
  const isAnalyzing = processingPhase === 'analysis';
  const isExporting = processingPhase === 'export';
  const reviewVisible = isAnalyzing || previewItems.length > 0;
  const analysisComplete =
    sourcePaths.length > 0 && previewItems.length === sourcePaths.length && !isAnalyzing && !previewsStale;
  const previewByPath = useMemo(() => new Map(previewItems.map((item) => [item.sourcePath, item])), [previewItems]);
  const activePreview = activePreviewPath ? (previewByPath.get(activePreviewPath) ?? null) : (previewItems[0] ?? null);
  const progressValue =
    progress && progress.current > 0 && progress.total > 0
      ? Math.min(100, Math.max(0, (progress.current / progress.total) * 100))
      : null;
  const progressTitle = isAnalyzing
    ? t('modals.batchGeometry.analyzing')
    : isExporting
      ? t('modals.batchGeometry.processingTitle')
      : t('modals.batchGeometry.orientation');

  const resetReview = useCallback(() => {
    previewRefreshRequestIdRef.current += 1;
    setPreviewItems([]);
    setActivePreviewPath(null);
    setRefreshingPreviewPath(null);
    setPreviewsStale(false);
    setProgress(null);
    setResult(null);
  }, []);

  const appendPreviewItem = useCallback((item: BatchGeometryPreviewItem) => {
    startTransition(() => {
      setPreviewItems((currentItems) => {
        const existingIndex = currentItems.findIndex((currentItem) => currentItem.sourcePath === item.sourcePath);
        if (existingIndex < 0) return [...currentItems, item];
        const nextItems = currentItems.slice();
        nextItems[existingIndex] = item;
        return nextItems;
      });
    });
    setActivePreviewPath(item.sourcePath);
  }, []);

  useEffect(() => {
    if (!isOpen) return;

    const listeners = [
      listen<BatchGeometryProgress>('batch-geometry-progress', (event) => {
        setProgress({ ...event.payload, phase: 'export' });
      }).catch(() => null),
      listen<BatchGeometryProgress>('batch-geometry-orientation-progress', (event) => {
        setProgress({ ...event.payload, phase: 'analysis' });
      }).catch(() => null),
      listen<BatchGeometryPreviewItem>('batch-geometry-preview', (event) => {
        appendPreviewItem(event.payload);
      }).catch(() => null),
    ];
    return () => {
      void Promise.all(listeners).then((disposers) => {
        disposers.forEach((dispose) => {
          if (dispose) disposeTauriListener(dispose);
        });
      });
    };
  }, [appendPreviewItem, isOpen]);

  useEffect(() => {
    if (!isOpen) return;

    setSourcePaths([]);
    setOutputFolder('');
    setOutputFormat('jpg');
    setJpegQuality(92);
    setApplyDistortion(true);
    setProcessingPhase('idle');
    setProgress(null);
    setPreviewItems([]);
    setActivePreviewPath(null);
    setRefreshingPreviewPath(null);
    setShowGuides(false);
    setPreviewsStale(false);
    setResult(null);
    setError(null);
    closeButtonRef.current?.focus();
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || isProcessing) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isProcessing, onClose]);

  const handleClose = useCallback(() => {
    if (!isProcessing) onClose();
  }, [isProcessing, onClose]);

  const handlePickImages = useCallback(async () => {
    try {
      const supportedTypes = await getSupportedImageTypes();
      const rawExtensions = normalizeExtensions(supportedTypes.raw);
      const nonRawExtensions = normalizeExtensions(supportedTypes.nonRaw);
      const allExtensions = Array.from(new Set([...rawExtensions, ...nonRawExtensions]));
      const selected = await open({
        defaultPath: await homeDir(),
        filters: [
          { name: t('library.import.allSupportedImages'), extensions: allExtensions },
          { name: 'RAW', extensions: rawExtensions },
          { name: 'JPEG / PNG / TIFF / PSD', extensions: nonRawExtensions },
        ],
        multiple: true,
        title: t('modals.batchGeometry.selectImages'),
      });
      const selectedPaths = pathsFromSelection(selected);
      if (selectedPaths.length === 0) return;

      setSourcePaths((currentPaths) => Array.from(new Set([...currentPaths, ...selectedPaths])));
      resetReview();
      setError(null);
    } catch (pickError) {
      setError(errorMessage(pickError));
    }
  }, [resetReview, t]);

  const handleClearSources = useCallback(() => {
    setSourcePaths([]);
    resetReview();
    setError(null);
  }, [resetReview]);

  const handlePickOutputFolder = useCallback(async () => {
    try {
      const selected = await open({
        defaultPath: outputFolder || (await homeDir()),
        directory: true,
        multiple: false,
        title: t('modals.batchGeometry.outputFolder'),
      });
      if (typeof selected === 'string') {
        setOutputFolder(selected);
        setError(null);
        setResult(null);
      }
    } catch (pickError) {
      setError(errorMessage(pickError));
    }
  }, [outputFolder, t]);

  const handleAnalyze = useCallback(async () => {
    if (sourcePaths.length === 0) {
      setError(t('modals.batchGeometry.noImages'));
      return;
    }

    setError(null);
    resetReview();
    setProcessingPhase('analysis');
    try {
      const analysis = await invoke<BatchGeometryAnalysisResult>(Invokes.AnalyzeBatchGeometry, {
        applyDistortion,
        paths: sourcePaths,
        showGuides,
        useContentOrientation: true,
      });
      startTransition(() => setPreviewItems(analysis.items));
      setPreviewsStale(false);
      setActivePreviewPath((currentPath) => currentPath ?? analysis.items[0]?.sourcePath ?? null);
    } catch (analysisError) {
      setError(errorMessage(analysisError));
    } finally {
      setProcessingPhase('idle');
    }
  }, [applyDistortion, resetReview, showGuides, sourcePaths, t]);

  const refreshPreviewItems = useCallback(
    async (nextApplyDistortion: boolean, nextShowGuides: boolean) => {
      if (previewItems.length === 0) return;

      const requestId = ++previewRefreshRequestIdRef.current;
      const itemsToRefresh = previewItems.slice();
      setPreviewsStale(true);
      setProcessingPhase('analysis');
      setProgress({ current: 0, path: '', phase: 'analysis', total: itemsToRefresh.length });
      setError(null);

      try {
        for (const [index, item] of itemsToRefresh.entries()) {
          if (requestId !== previewRefreshRequestIdRef.current) return;
          setRefreshingPreviewPath(item.sourcePath);
          setProgress({ current: index, path: item.sourcePath, phase: 'analysis', total: itemsToRefresh.length });

          const refreshedPreview = await invoke<BatchGeometryPreviewItem>(Invokes.PreviewBatchGeometryCorrection, {
            applyDistortion: nextApplyDistortion,
            orientationSteps: item.orientationSteps,
            path: item.sourcePath,
            showGuides: nextShowGuides,
          });
          if (requestId !== previewRefreshRequestIdRef.current) return;

          setPreviewItems((currentItems) =>
            currentItems.map((currentItem) =>
              currentItem.sourcePath === item.sourcePath
                ? {
                    ...currentItem,
                    afterPreview: refreshedPreview.afterPreview,
                    beforePreview: refreshedPreview.beforePreview,
                    distortionApplied: refreshedPreview.distortionApplied,
                  }
                : currentItem,
            ),
          );
          setProgress({
            current: index + 1,
            path: item.sourcePath,
            phase: 'analysis',
            total: itemsToRefresh.length,
          });
        }
        if (requestId === previewRefreshRequestIdRef.current) setPreviewsStale(false);
      } catch (previewError) {
        if (requestId === previewRefreshRequestIdRef.current) {
          setError(errorMessage(previewError));
          setPreviewsStale(true);
        }
      } finally {
        if (requestId === previewRefreshRequestIdRef.current) {
          setRefreshingPreviewPath(null);
          setProcessingPhase('idle');
          setProgress(null);
        }
      }
    },
    [previewItems],
  );

  const handleDistortionChange = useCallback(
    (nextApplyDistortion: boolean) => {
      setApplyDistortion(nextApplyDistortion);
      if (previewItems.length > 0) void refreshPreviewItems(nextApplyDistortion, showGuides);
      setError(null);
    },
    [previewItems.length, refreshPreviewItems, showGuides],
  );

  const handleGuidesChange = useCallback(() => {
    const nextShowGuides = !showGuides;
    setShowGuides(nextShowGuides);
    if (previewItems.length > 0) void refreshPreviewItems(applyDistortion, nextShowGuides);
  }, [applyDistortion, previewItems.length, refreshPreviewItems, showGuides]);

  const handlePreviewOrientation = useCallback(
    async (nextOrientationSteps: number) => {
      if (!activePreview || isProcessing) return;
      const orientationSteps = ((nextOrientationSteps % 4) + 4) % 4;
      const requestId = ++previewRefreshRequestIdRef.current;
      setError(null);
      setRefreshingPreviewPath(activePreview.sourcePath);
      setPreviewsStale(true);
      setProcessingPhase('analysis');
      setProgress({ current: 0, path: activePreview.sourcePath, phase: 'analysis', total: 1 });
      try {
        const refreshedPreview = await invoke<BatchGeometryPreviewItem>(Invokes.PreviewBatchGeometryCorrection, {
          applyDistortion,
          orientationSteps,
          path: activePreview.sourcePath,
          showGuides,
        });
        if (requestId !== previewRefreshRequestIdRef.current) return;
        setPreviewItems((currentItems) =>
          currentItems.map((item) =>
            item.sourcePath === activePreview.sourcePath
              ? {
                  ...item,
                  afterPreview: refreshedPreview.afterPreview,
                  beforePreview: refreshedPreview.beforePreview,
                  distortionApplied: refreshedPreview.distortionApplied,
                  orientationSteps,
                }
              : item,
          ),
        );
        setPreviewsStale(false);
      } catch (previewError) {
        if (requestId === previewRefreshRequestIdRef.current) {
          setError(errorMessage(previewError));
          setPreviewsStale(true);
        }
      } finally {
        if (requestId === previewRefreshRequestIdRef.current) {
          setRefreshingPreviewPath(null);
          setProcessingPhase('idle');
          setProgress(null);
        }
      }
    },
    [activePreview, applyDistortion, isProcessing, showGuides],
  );

  const handleExport = useCallback(async () => {
    if (!outputFolder) {
      setError(t('modals.batchGeometry.noOutputFolder'));
      return;
    }
    if (!analysisComplete) {
      setError(t('modals.batchGeometry.previewNotReady'));
      return;
    }

    setError(null);
    setResult(null);
    setProgress(null);
    setProcessingPhase('export');
    try {
      const correctionResult = await invoke<BatchGeometryCorrectionResult>(Invokes.BatchGeometryCorrection, {
        applyDistortion,
        jpegQuality,
        orientationOverrides: previewItems.map((item) => ({
          orientationSteps: item.orientationSteps,
          path: item.sourcePath,
        })),
        outputFolder,
        outputFormat,
        paths: sourcePaths,
        useContentOrientation: false,
      });
      setResult(correctionResult);
    } catch (correctionError) {
      setError(errorMessage(correctionError));
    } finally {
      setProcessingPhase('idle');
    }
  }, [analysisComplete, applyDistortion, jpegQuality, outputFolder, outputFormat, previewItems, sourcePaths, t]);

  const handleRevealResult = useCallback(async () => {
    if (!result?.outputFolder) return;
    try {
      await invoke(Invokes.ShowInFinder, { path: result.outputFolder });
    } catch (revealError) {
      setError(errorMessage(revealError));
    }
  }, [result?.outputFolder]);

  const orientationLabel = useCallback((steps: number) => t(ORIENTATION_KEYS[normalizeOrientationSteps(steps)]), [t]);

  const handleBackdropPointerDown = (event: React.MouseEvent<HTMLDivElement>) => {
    backdropPointerTarget.current = event.target;
  };

  const handleBackdropClick = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.target === event.currentTarget && backdropPointerTarget.current === event.currentTarget) {
      handleClose();
    }
    backdropPointerTarget.current = null;
  };

  if (!isOpen) return null;

  return (
    <div className="app-modal-backdrop" onClick={handleBackdropClick} onMouseDown={handleBackdropPointerDown}>
      <section
        aria-describedby="batch-geometry-description"
        aria-labelledby="batch-geometry-title"
        aria-modal="true"
        className="app-modal-surface app-modal-surface--structured batch-geometry-modal-surface"
        role="dialog"
      >
        <header className="flex items-start justify-between gap-4 border-b border-border-color px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <ScanLine aria-hidden="true" size={18} />
            </div>
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-text-primary" id="batch-geometry-title">
                {t('modals.batchGeometry.title')}
              </h2>
              <p className="mt-1 text-xs leading-5 text-text-secondary" id="batch-geometry-description">
                {t('modals.batchGeometry.description')}
              </p>
            </div>
          </div>
          <button
            aria-label={t('modals.batchGeometry.close')}
            className="ui-icon-button ui-icon-button--md"
            disabled={isProcessing}
            onClick={handleClose}
            ref={closeButtonRef}
            type="button"
          >
            <X aria-hidden="true" size={17} />
          </button>
        </header>

        <div className="batch-geometry-modal-body">
          {result ? (
            <div className="rounded-lg border border-status-success/30 bg-status-success/8 p-4">
              <div className="flex items-start gap-3">
                <CheckCircle2 aria-hidden="true" className="mt-0.5 shrink-0 text-status-success" size={20} />
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold text-text-primary">{t('modals.batchGeometry.resultTitle')}</h3>
                  <p className="mt-1 text-xs leading-5 text-text-secondary">
                    {t('modals.batchGeometry.successDescription')}
                  </p>
                </div>
              </div>
              <dl className="mt-4 grid grid-cols-1 gap-2 text-xs text-text-secondary sm:grid-cols-2">
                <div className="rounded-md bg-bg-primary/70 px-3 py-2">
                  <dt className="sr-only">
                    {t('modals.batchGeometry.selectedCount', { total: result.processedCount })}
                  </dt>
                  <dd>{t('modals.batchGeometry.selectedCount', { total: result.processedCount })}</dd>
                </div>
                <div className="rounded-md bg-bg-primary/70 px-3 py-2">
                  <dt className="sr-only">
                    {t('modals.batchGeometry.resultOrientation', { total: result.contentOrientationCorrectedCount })}
                  </dt>
                  <dd>
                    {t('modals.batchGeometry.resultOrientation', { total: result.contentOrientationCorrectedCount })}
                  </dd>
                </div>
                {applyDistortion && (
                  <>
                    <div className="rounded-md bg-bg-primary/70 px-3 py-2">
                      <dt className="sr-only">
                        {t('modals.batchGeometry.resultDistortion', { total: result.distortionCorrectedCount })}
                      </dt>
                      <dd>{t('modals.batchGeometry.resultDistortion', { total: result.distortionCorrectedCount })}</dd>
                    </div>
                    <div className="rounded-md bg-bg-primary/70 px-3 py-2">
                      <dt className="sr-only">
                        {t('modals.batchGeometry.resultMissingProfile', {
                          total: result.missingDistortionProfileCount,
                        })}
                      </dt>
                      <dd>
                        {t('modals.batchGeometry.resultMissingProfile', {
                          total: result.missingDistortionProfileCount,
                        })}
                      </dd>
                    </div>
                  </>
                )}
              </dl>
              <p className="mt-3 truncate rounded-md bg-bg-primary/70 px-3 py-2 font-mono text-[11px] text-text-secondary">
                {result.outputFolder}
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <section aria-labelledby="batch-geometry-sources-title">
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="text-sm font-medium text-text-primary" id="batch-geometry-sources-title">
                      {t('modals.batchGeometry.sourceFiles')}
                    </h3>
                    <p className="mt-0.5 text-xs text-text-secondary">
                      {t('modals.batchGeometry.selectedCount', { total: sourcePaths.length })}
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    {sourcePaths.length > 0 && (
                      <>
                        <Button
                          disabled={isProcessing}
                          onClick={handleClearSources}
                          size="sm"
                          type="button"
                          variant="ghost"
                        >
                          {t('modals.batchGeometry.clear')}
                        </Button>
                        <Button
                          disabled={isProcessing}
                          onClick={handlePickImages}
                          size="sm"
                          type="button"
                          variant="secondary"
                        >
                          <ImagePlus aria-hidden="true" size={16} />
                          {t('modals.batchGeometry.addImages')}
                        </Button>
                      </>
                    )}
                  </div>
                </div>
                {sourcePaths.length > 0 && !reviewVisible ? (
                  <ul className="mt-3 space-y-1 rounded-lg border border-border-color bg-bg-primary/45 p-2">
                    {sourcePaths.slice(0, 3).map((path) => (
                      <li className="flex min-w-0 items-center gap-2 px-2 py-1 text-xs text-text-secondary" key={path}>
                        <ImagePlus aria-hidden="true" className="shrink-0 text-text-tertiary" size={14} />
                        <span className="truncate" title={path}>
                          {displayFileName(path)}
                        </span>
                      </li>
                    ))}
                    {sourcePaths.length > 3 && (
                      <li className="px-2 py-1 text-xs text-text-tertiary">+{sourcePaths.length - 3}</li>
                    )}
                  </ul>
                ) : sourcePaths.length === 0 ? (
                  <div className="mt-3 rounded-lg border border-dashed border-border-color bg-bg-primary/30 transition-colors hover:border-accent/60 hover:bg-card-active">
                    <button
                      className="flex w-full items-center justify-center gap-2 rounded-lg bg-transparent px-4 py-5 text-xs text-text-secondary transition-colors hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                      disabled={isProcessing}
                      onClick={handlePickImages}
                      type="button"
                    >
                      <ImagePlus aria-hidden="true" size={17} />
                      {t('modals.batchGeometry.selectImages')}
                    </button>
                  </div>
                ) : null}
              </section>

              {reviewVisible && (
                <BatchGeometryComparison
                  activeItem={activePreview}
                  activePath={activePreview?.sourcePath ?? null}
                  applyDistortion={applyDistortion}
                  isProcessing={isProcessing}
                  isRefreshing={refreshingPreviewPath === activePreview?.sourcePath}
                  items={previewItems}
                  onChoose={setActivePreviewPath}
                  onRestore={() =>
                    activePreview && void handlePreviewOrientation(activePreview.suggestedOrientationSteps)
                  }
                  onRotate={(delta) =>
                    activePreview && void handlePreviewOrientation(activePreview.orientationSteps + delta)
                  }
                  onToggleGuides={handleGuidesChange}
                  orientationLabel={orientationLabel}
                  showGuides={showGuides}
                />
              )}

              <section aria-labelledby="batch-geometry-destination-title" className="border-t border-border-color pt-4">
                <h3 className="text-sm font-medium text-text-primary" id="batch-geometry-destination-title">
                  {t('modals.batchGeometry.destination')}
                </h3>
                <div className="mt-2 flex min-w-0 items-center gap-2">
                  <Button
                    className="shrink-0"
                    disabled={isProcessing}
                    onClick={handlePickOutputFolder}
                    size="sm"
                    type="button"
                    variant="secondary"
                  >
                    <FolderOpen aria-hidden="true" size={16} />
                    {t('modals.batchGeometry.outputFolder')}
                  </Button>
                  <p className="min-w-0 truncate text-xs text-text-secondary" title={outputFolder || undefined}>
                    {outputFolder || t('modals.batchGeometry.noOutputFolder')}
                  </p>
                </div>
              </section>

              <section aria-labelledby="batch-geometry-options-title" className="border-t border-border-color pt-4">
                <h3 className="sr-only" id="batch-geometry-options-title">
                  {t('modals.batchGeometry.title')}
                </h3>
                <div className="grid gap-4 sm:grid-cols-2">
                  <label className="block min-w-0 text-xs font-medium text-text-secondary">
                    {t('modals.batchGeometry.format')}
                    <Dropdown
                      ariaLabel={t('modals.batchGeometry.format')}
                      className="mt-1.5"
                      disabled={isProcessing}
                      onChange={setOutputFormat}
                      options={outputFormatOptions}
                      value={outputFormat}
                    />
                  </label>
                  {outputFormat === 'jpg' && (
                    <div className="min-w-0 pt-0.5">
                      <Slider
                        defaultValue={92}
                        disabled={isProcessing}
                        fillOrigin="min"
                        label={t('modals.batchGeometry.jpegQuality')}
                        max={100}
                        min={1}
                        onChange={(event) => setJpegQuality(Number(event.target.value))}
                        showPositiveSign={false}
                        step={1}
                        value={jpegQuality}
                      />
                    </div>
                  )}
                </div>
                <div className="mt-4 space-y-3 rounded-lg bg-bg-primary/45 p-3">
                  <Switch
                    checked={applyDistortion}
                    disabled={isProcessing}
                    id="batch-geometry-distortion"
                    label={t('modals.batchGeometry.applyDistortion')}
                    onChange={handleDistortionChange}
                  />
                  <div className="flex gap-2 border-t border-border-color pt-3 text-xs leading-5 text-text-secondary">
                    <Eye aria-hidden="true" className="mt-0.5 shrink-0 text-accent" size={15} />
                    <div>
                      <p className="font-medium text-text-primary">{t('modals.batchGeometry.orientation')}</p>
                      <p className="mt-0.5">{t('modals.batchGeometry.orientationDescription')}</p>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}

          {error && (
            <div
              aria-live="assertive"
              className="mt-4 rounded-lg border border-status-error/40 bg-status-error/10 px-3 py-2 text-xs leading-5 text-status-error"
              role="alert"
            >
              <p className="font-medium">{t('modals.batchGeometry.errorTitle')}</p>
              <p className="mt-0.5 break-words text-text-secondary">{error}</p>
            </div>
          )}

          {isProcessing && (
            <div className="mt-4 rounded-lg border border-border-color bg-bg-primary/45 p-3">
              <div className="flex items-center gap-2 text-sm font-medium text-text-primary">
                <Loader2 aria-hidden="true" className="animate-spin text-accent" size={16} />
                {progressTitle}
              </div>
              <TaskProgress
                ariaLabel={progressTitle}
                className="mt-3"
                indeterminate={progressValue === null}
                label={
                  progress && progress.current > 0
                    ? t('modals.batchGeometry.processing', { current: progress.current, total: progress.total })
                    : progressTitle
                }
                showPercentage={false}
                value={progressValue}
              />
              {progress?.path && (
                <p className="mt-2 truncate text-[11px] text-text-tertiary" title={progress.path}>
                  {displayFileName(progress.path)}
                </p>
              )}
            </div>
          )}
        </div>

        <footer className="app-modal-footer app-modal-footer--inset px-5 pb-4">
          {result ? (
            <>
              <Button onClick={handleClose} type="button" variant="secondary">
                {t('modals.batchGeometry.close')}
              </Button>
              <Button onClick={handleRevealResult} type="button">
                <FolderOpen aria-hidden="true" size={16} />
                {t('modals.batchGeometry.reveal')}
              </Button>
            </>
          ) : (
            <>
              <Button disabled={isProcessing} onClick={handleClose} type="button" variant="ghost">
                {t('modals.batchGeometry.cancel')}
              </Button>
              {analysisComplete ? (
                <div className="flex items-center gap-2">
                  <Button disabled={isProcessing} onClick={handleAnalyze} size="sm" type="button" variant="secondary">
                    <ScanLine aria-hidden="true" size={15} />
                    {t('modals.batchGeometry.reAnalyze')}
                  </Button>
                  <Button disabled={isProcessing || !outputFolder} onClick={handleExport} type="button">
                    {isExporting ? (
                      <Loader2 aria-hidden="true" className="animate-spin" size={16} />
                    ) : (
                      <CheckCircle2 aria-hidden="true" size={16} />
                    )}
                    {t('modals.batchGeometry.exportReviewed')}
                  </Button>
                </div>
              ) : (
                <Button disabled={isProcessing || sourcePaths.length === 0} onClick={handleAnalyze} type="button">
                  {isAnalyzing ? (
                    <Loader2 aria-hidden="true" className="animate-spin" size={16} />
                  ) : (
                    <Eye aria-hidden="true" size={16} />
                  )}
                  {t('modals.batchGeometry.analyze')}
                </Button>
              )}
            </>
          )}
        </footer>
      </section>
    </div>
  );
}
