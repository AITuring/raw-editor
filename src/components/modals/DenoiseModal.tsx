import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle, XCircle, Loader2, Save, RefreshCw, ZoomIn, ZoomOut, Move, Grip } from 'lucide-react';
import { motion } from 'framer-motion';
import Button from '../ui/Button';
import Dropdown from '../ui/Dropdown';
import Slider from '../ui/Slider';
import Text from '../ui/Text';
import TaskProgress from '../ui/TaskProgress';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { listen } from '@tauri-apps/api/event';
import { getMessageTaskProgress } from '../../utils/taskProgress';
import { disposeTauriListener } from '../../utils/tauriListenerCleanup';

interface DenoiseModalProps {
  isOpen: boolean;
  onClose(): void;
  onDenoise(intensity: number, method: 'ai' | 'bm3d'): void;
  onBatchDenoise(intensity: number, method: 'ai' | 'bm3d', paths: string[]): Promise<string[]>;
  onSave(): Promise<string>;
  onOpenFile(path: string): void;
  error: string | null;
  previewBase64: string | null;
  originalBase64: string | null;
  isProcessing: boolean;
  progressMessage: string | null;
  aiModelDownloadStatus: string | null;
  isRaw: boolean;
  loadingImageUrl?: string | null;
  targetPaths: string[];
}

const ImageCompare = ({ original, denoised }: { original: string; denoised: string }) => {
  const { t } = useTranslation();
  const [sliderPosition, setSliderPosition] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  const [isDragging, setIsDragging] = useState(false);
  const [isResizingSlider, setIsResizingSlider] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const lastMousePos = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!isDragging && !isResizingSlider) return;

    const handleWindowMouseMove = (e: MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;

      if (isResizingSlider) {
        const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
        const percent = (x / rect.width) * 100;
        setSliderPosition(percent);
      } else if (isDragging) {
        const dx = e.clientX - lastMousePos.current.x;
        const dy = e.clientY - lastMousePos.current.y;
        setPan((prev) => ({ x: prev.x + dx, y: prev.y + dy }));
        lastMousePos.current = { x: e.clientX, y: e.clientY };
      }
    };

    const handleWindowMouseUp = () => {
      setIsDragging(false);
      setIsResizingSlider(false);
    };

    window.addEventListener('mousemove', handleWindowMouseMove);
    window.addEventListener('mouseup', handleWindowMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleWindowMouseMove);
      window.removeEventListener('mouseup', handleWindowMouseUp);
    };
  }, [isDragging, isResizingSlider]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isResizingSlider) return;
    e.preventDefault();
    setIsDragging(true);
    lastMousePos.current = { x: e.clientX, y: e.clientY };
  };

  const handleSliderMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingSlider(true);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left - rect.width / 2;
    const mouseY = e.clientY - rect.top - rect.height / 2;

    const delta = -e.deltaY * 0.001;
    const newZoom = Math.min(Math.max(0.5, zoom + delta), 4);

    const scaleRatio = newZoom / zoom;
    const mouseFromCenterX = mouseX - pan.x;
    const mouseFromCenterY = mouseY - pan.y;

    const newPanX = mouseX - mouseFromCenterX * scaleRatio;
    const newPanY = mouseY - mouseFromCenterY * scaleRatio;

    setZoom(newZoom);
    setPan({ x: newPanX, y: newPanY });
  };

  const imageTransformStyle = {
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    transition: isDragging || isResizingSlider ? 'none' : 'transform 0.1s ease-out',
    transformOrigin: 'center center',
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-lg border border-border-color bg-[#111]">
      <div className="flex h-9 items-center justify-between border-b border-border-color bg-bg-primary px-3">
        <Text as="div" variant={TextVariants.small} className="flex items-center gap-2">
          <Move size={14} /> <span>{t('modals.denoise.panZoomEnabled')}</span>
        </Text>
        <Text as="div" variant={TextVariants.small} className="flex items-center gap-2">
          <button onClick={() => setZoom((z) => Math.max(0.5, z - 0.5))} className="hover:text-text-primary">
            <ZoomOut size={16} />
          </button>
          <span className="w-10 text-center">{(zoom * 100).toFixed(0)}%</span>
          <button onClick={() => setZoom((z) => Math.min(4, z + 0.5))} className="hover:text-text-primary">
            <ZoomIn size={16} />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              setPan({ x: 0, y: 0 });
              setSliderPosition(50);
            }}
            className="ui-inline-action ml-1"
          >
            {t('modals.denoise.reset')}
          </button>
        </Text>
      </div>

      <div
        ref={containerRef}
        className="flex-1 relative overflow-hidden cursor-grab active:cursor-grabbing select-none"
        onMouseDown={handleMouseDown}
        onWheel={handleWheel}
      >
        <div className="absolute inset-0 flex items-center justify-center overflow-hidden pointer-events-none">
          <div className="origin-center" style={imageTransformStyle}>
            <img
              src={denoised}
              alt="Denoised"
              className="max-w-none shadow-xl"
              style={{ height: 'auto' }}
              draggable={false}
            />
          </div>
        </div>

        <div
          className="absolute inset-0 flex items-center justify-center overflow-hidden pointer-events-none"
          style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        >
          <div className="origin-center" style={imageTransformStyle}>
            <img
              src={original}
              alt="Original"
              className="max-w-none shadow-xl"
              style={{ height: 'auto' }}
              draggable={false}
            />
          </div>
        </div>

        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white cursor-col-resize z-10 shadow-[0_0_8px_rgba(0,0,0,0.8)]"
          style={{ left: `${sliderPosition}%` }}
          onMouseDown={handleSliderMouseDown}
        >
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center gap-0.5">
            <div className="w-0.5 h-3 bg-black/40 rounded-full"></div>
            <div className="w-0.5 h-3 bg-black/40 rounded-full"></div>
          </div>
        </div>

        <Text
          as="div"
          variant={TextVariants.small}
          color={TextColors.white}
          weight={TextWeights.medium}
          className="absolute top-3 left-3 bg-black/60 backdrop-blur-xs px-2.5 py-1 rounded-md pointer-events-none z-0"
        >
          {t('modals.denoise.original')}
        </Text>
        <Text
          as="div"
          variant={TextVariants.small}
          color={TextColors.button}
          weight={TextWeights.medium}
          className="absolute top-3 right-3 bg-accent/90 backdrop-blur-xs px-2.5 py-1 rounded-md pointer-events-none z-0"
        >
          {t('modals.denoise.denoised')}
        </Text>
      </div>
    </div>
  );
};

export default function DenoiseModal({
  isOpen,
  onClose,
  onDenoise,
  onBatchDenoise,
  onSave,
  onOpenFile,
  error,
  previewBase64,
  originalBase64,
  isProcessing,
  progressMessage,
  aiModelDownloadStatus,
  isRaw,
  loadingImageUrl,
  targetPaths,
}: DenoiseModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [intensity, setIntensity] = useState<number>(15);
  const [method, setMethod] = useState<'ai' | 'bm3d'>('ai');
  const [isSaving, setIsSaving] = useState(false);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [lastRunSettings, setLastRunSettings] = useState<{
    intensity: number;
    method: 'ai' | 'bm3d';
  } | null>(null);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; path: string } | null>(null);
  const isBatch = targetPaths.length > 1;
  const mouseDownTarget = useRef<EventTarget | null>(null);

  const methodOptions = useMemo<Array<{ label: string; value: 'ai' | 'bm3d' }>>(
    () => [
      { label: t('modals.denoise.methodAi'), value: 'ai' },
      { label: t('modals.denoise.methodBm3d'), value: 'bm3d' },
    ],
    [t],
  );

  useEffect(() => {
    const unlisten = listen('denoise-batch-progress', (e: any) => {
      setBatchProgress(e.payload);
    });
    return () => {
      void unlisten.then((dispose) => disposeTauriListener(dispose));
    };
  }, []);

  const currentStatusText =
    isBatch && batchProgress
      ? t('modals.denoise.batchProgressText', { current: batchProgress.current, total: batchProgress.total })
      : aiModelDownloadStatus?.includes('NIND')
        ? t('modals.denoise.downloadingText', { status: aiModelDownloadStatus })
        : progressMessage || t('modals.denoise.initializing');
  const denoiseProgress = useMemo(
    () => getMessageTaskProgress(isBatch ? null : currentStatusText, 'denoise'),
    [currentStatusText, isBatch],
  );
  const batchOverallProgress =
    batchProgress && batchProgress.total > 0 ? (batchProgress.current / batchProgress.total) * 100 : null;
  const previewNeedsUpdate = Boolean(
    !isBatch &&
    previewBase64 &&
    lastRunSettings &&
    (lastRunSettings.intensity !== intensity || lastRunSettings.method !== method),
  );

  useEffect(() => {
    if (isOpen) {
      setMethod(isRaw ? 'ai' : 'bm3d');
      setIntensity(isRaw ? 50 : 15);
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
        setSavedPath(null);
        setIsSaving(false);
        setLastRunSettings(null);
        setBatchProgress(null);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen, isRaw]);

  const handleClose = useCallback(() => {
    if (isSaving) return;
    onClose();
  }, [onClose, isSaving]);

  const handleBackdropMouseDown = (e: React.MouseEvent) => {
    mouseDownTarget.current = e.target;
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && mouseDownTarget.current === e.currentTarget) {
      handleClose();
    }
    mouseDownTarget.current = null;
  };

  const handleRunDenoise = async () => {
    setSavedPath(null);
    if (isBatch) {
      setIsSaving(true);
      setBatchProgress({ current: 0, total: targetPaths.length, path: targetPaths[0] ?? '' });
      try {
        await onBatchDenoise(intensity / 100, method, targetPaths);
        onClose();
      } catch (e) {
        console.error('Batch denoise failed:', e);
      } finally {
        setIsSaving(false);
        setBatchProgress(null);
      }
    } else {
      setLastRunSettings({ intensity, method });
      onDenoise(intensity / 100, method);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const path = await onSave();
      setSavedPath(path);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSaving(false);
    }
  };

  const handleOpen = () => {
    if (savedPath) {
      onOpenFile(savedPath);
      handleClose();
    }
  };

  const renderContent = () => {
    if (error) {
      return (
        <div className="denoise-modal-stage denoise-modal-stage--message flex flex-col items-center justify-center py-10">
          <div className="flex items-center justify-center mb-6">
            <XCircle className="w-12 h-12 text-status-error" />
          </div>
          <Text variant={TextVariants.title} className="mb-2 text-center">
            {t('modals.denoise.processingFailed')}
          </Text>
          <Text className="text-center p-4 rounded-lg bg-bg-primary max-w-md mt-2 leading-relaxed">
            {String(error)}
          </Text>
        </div>
      );
    }

    if (previewBase64 && originalBase64 && !isProcessing && !isBatch) {
      return (
        <div className="denoise-modal-stage denoise-modal-stage--preview w-full">
          <ImageCompare original={originalBase64} denoised={previewBase64} />
          {savedPath && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
              <Text
                as="div"
                variant={TextVariants.heading}
                color={TextColors.success}
                className="flex items-center justify-center gap-2 mt-4"
              >
                <CheckCircle className="w-5 h-5" />
                <span>{t('modals.denoise.saveSuccess')}</span>
              </Text>
            </motion.div>
          )}
        </div>
      );
    }

    if (isProcessing || (isBatch && isSaving)) {
      return (
        <div className="denoise-modal-stage denoise-modal-stage--processing flex overflow-hidden rounded-lg border border-border-color">
          <div className="w-2/5 relative overflow-hidden shrink-0 bg-[#0a0a0a] flex items-center justify-center">
            {loadingImageUrl ? (
              <img src={loadingImageUrl} alt="Selected preview" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full bg-surface/50" />
            )}
          </div>
          <div className="flex-1 flex flex-col items-center justify-center px-12 bg-bg-primary">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.4 }}
              className="flex flex-col items-center w-full"
            >
              <Text variant={TextVariants.title} className="mb-2 text-center">
                {t('modals.denoise.denoisingProgress')}
              </Text>
              <TaskProgress
                ariaLabel={t('modals.denoise.denoisingProgress')}
                className="mt-5 max-w-sm"
                indeterminate={isBatch ? batchOverallProgress === null : denoiseProgress.value === null}
                label={currentStatusText}
                showPercentage={isBatch || denoiseProgress.exact}
                value={isBatch ? batchOverallProgress : denoiseProgress.value}
              />

              <Text
                variant={TextVariants.small}
                data-tooltip={t('modals.denoise.gpuWarningTooltip')}
                className="mt-6 text-center max-w-xs opacity-60"
              >
                {t('modals.denoise.speedNotice')}
              </Text>
            </motion.div>
          </div>
        </div>
      );
    }

    return (
      <div className="denoise-modal-stage denoise-modal-stage--idle flex flex-col items-center justify-center">
        <div className="flex items-center justify-center mb-6">
          <Grip className="w-12 h-12 text-accent" />
        </div>
        <Text variant={TextVariants.title} className="mb-3 text-center">
          {isBatch ? t('modals.denoise.titleBatch') : t('modals.denoise.titleSingle')}
        </Text>
        <Text className="text-center max-w-md leading-relaxed">{t('modals.denoise.description')}</Text>
      </div>
    );
  };

  const renderButtons = () => {
    if (error) {
      return (
        <Button onClick={handleClose} className="w-full">
          {t('modals.denoise.close')}
        </Button>
      );
    }

    if (savedPath) {
      return (
        <>
          <Button onClick={handleClose} variant="secondary">
            {t('modals.denoise.close')}
          </Button>
          <Button onClick={handleOpen}>{t('modals.denoise.openInEditor')}</Button>
        </>
      );
    }

    const disabled = isProcessing || isSaving;

    return (
      <div className={`denoise-modal-controls ${disabled ? 'is-disabled' : ''}`}>
        <div className="denoise-modal-settings">
          <div className="denoise-modal-field denoise-modal-method-field">
            <Text variant={TextVariants.body} weight={TextWeights.medium}>
              {t('modals.denoise.methodLabel')}
            </Text>
            <Dropdown
              options={methodOptions}
              value={method}
              onChange={(val) => {
                setMethod(val);
                setIntensity(val === 'ai' ? 50 : 15);
              }}
            />
          </div>
          <div className="denoise-modal-field denoise-modal-intensity-field">
            <Slider
              label={method === 'ai' ? t('modals.denoise.qualityTileSizeLabel') : t('modals.denoise.strengthLabel')}
              value={intensity}
              min={0}
              max={100}
              step={1}
              defaultValue={method === 'ai' ? 50 : 15}
              onChange={(e) => setIntensity(Number(e.target.value))}
              trackClassName="bg-bg-secondary"
              fillOrigin="min"
            />
            {previewNeedsUpdate && (
              <Text
                as="div"
                variant={TextVariants.small}
                color={TextColors.warning}
                className="semantic-status mt-2 leading-tight"
                data-tone="warning"
                aria-live="polite"
                role="status"
              >
                <RefreshCw aria-hidden="true" className="shrink-0" size={12} />
                <span>{t('modals.denoise.previewNeedsUpdate')}</span>
              </Text>
            )}
          </div>
        </div>

        <div className="denoise-modal-divider" aria-hidden="true" />

        <div className="denoise-modal-actions">
          <Button onClick={handleClose} variant="secondary">
            {previewBase64 ? t('modals.denoise.close') : t('modals.denoise.cancel')}
          </Button>

          <Button
            onClick={handleRunDenoise}
            disabled={isProcessing || isSaving}
            variant={previewBase64 && !isBatch && !previewNeedsUpdate ? 'secondary' : 'primary'}
          >
            {isProcessing || (isBatch && isSaving) ? (
              <Loader2 className="animate-spin mr-2" size={16} />
            ) : previewBase64 && !isBatch ? (
              <RefreshCw className="mr-2" size={16} />
            ) : (
              <Grip className="mr-2" size={16} />
            )}
            {isBatch
              ? t('modals.denoise.btnBatchDenoise')
              : previewBase64
                ? t('modals.denoise.btnRetry')
                : t('modals.denoise.btnStart')}
          </Button>

          {previewBase64 && !isBatch && (
            <Button onClick={handleSave} disabled={isSaving || isProcessing || previewNeedsUpdate}>
              {isSaving ? <Loader2 className="animate-spin mr-2" size={16} /> : <Save className="mr-2" size={16} />}
              {t('modals.denoise.btnSave')}
            </Button>
          )}
        </div>
      </div>
    );
  };

  if (!isMounted) return null;

  return (
    <div
      className={`app-modal-backdrop ${show ? 'opacity-100' : 'opacity-0'}`}
      onMouseDown={handleBackdropMouseDown}
      onClick={handleBackdropClick}
    >
      <div
        className={`app-modal-surface app-modal-surface--structured denoise-modal-surface ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
      >
        <div className="app-modal-content denoise-modal-content">
          <div className="denoise-modal-body">{renderContent()}</div>
          {isSaving && !isProcessing && !isBatch && (
            <TaskProgress
              ariaLabel={t('modals.denoise.btnSave')}
              className="denoise-modal-progress"
              compact
              indeterminate
              label={t('modals.denoise.btnSave')}
            />
          )}
          <div
            className={`app-modal-footer app-modal-footer--inset denoise-modal-footer ${savedPath ? 'is-result' : ''}`}
          >
            {renderButtons()}
          </div>
        </div>
      </div>
    </div>
  );
}
