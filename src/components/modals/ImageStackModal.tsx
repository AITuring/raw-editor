import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import {
  ArrowDown,
  ArrowUp,
  Check,
  Cylinder,
  Expand,
  Globe2,
  Layers3,
  Loader2,
  Move,
  RefreshCw,
  Save,
  Scan,
  Sparkles,
  X,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../ui/Button';
import TaskProgress from '../ui/TaskProgress';
import type { ImageStackAlignmentMode, ImageStackBlendMode } from '../../store/useUIStore';

interface ImageStackModalProps {
  error: string | null;
  finalImageBase64: string | null;
  isOpen: boolean;
  isProcessing: boolean;
  progressMessage: string | null;
  sourcePaths: string[];
  initialBlendMode: ImageStackBlendMode;
  initialAlignmentMode: ImageStackAlignmentMode;
  thumbnails: Record<string, string>;
  onClose(): void;
  onChange(): void;
  onOpenFile(path: string): void;
  onProcess(paths: string[], blendMode: ImageStackBlendMode, alignmentMode: ImageStackAlignmentMode): void;
  onSave(blendMode: ImageStackBlendMode): Promise<string>;
}

const getDisplayName = (path: string) => {
  const cleanPath = path.split('?')[0];
  return cleanPath.split(/[\\/]/).pop() || cleanPath;
};

const ALIGNMENT_OPTIONS: Array<{
  value: ImageStackAlignmentMode;
  icon: typeof Scan;
  labelKey: string;
  descriptionKey: string;
}> = [
  {
    value: 'auto',
    icon: Sparkles,
    labelKey: 'auto',
    descriptionKey: 'autoDescription',
  },
  {
    value: 'perspective',
    icon: Scan,
    labelKey: 'perspective',
    descriptionKey: 'perspectiveDescription',
  },
  {
    value: 'cylindrical',
    icon: Cylinder,
    labelKey: 'cylindrical',
    descriptionKey: 'cylindricalDescription',
  },
  {
    value: 'spherical',
    icon: Globe2,
    labelKey: 'spherical',
    descriptionKey: 'sphericalDescription',
  },
  {
    value: 'position',
    icon: Move,
    labelKey: 'position',
    descriptionKey: 'positionDescription',
  },
];

export default function ImageStackModal({
  error,
  finalImageBase64,
  isOpen,
  isProcessing,
  progressMessage,
  sourcePaths,
  initialBlendMode,
  initialAlignmentMode,
  thumbnails,
  onClose,
  onChange,
  onOpenFile,
  onProcess,
  onSave,
}: ImageStackModalProps) {
  const { t } = useTranslation();
  const [orderedPaths, setOrderedPaths] = useState<string[]>(sourcePaths);
  const [blendMode, setBlendMode] = useState<ImageStackBlendMode>(initialBlendMode);
  const [alignmentMode, setAlignmentMode] = useState<ImageStackAlignmentMode>(initialAlignmentMode);
  const [isSaving, setIsSaving] = useState(false);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const mouseDownTarget = useRef<EventTarget | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    if (!isOpen) return;
    setOrderedPaths(sourcePaths);
    setBlendMode(initialBlendMode);
    setAlignmentMode(initialAlignmentMode);
    setSavedPath(null);
    setIsMounted(true);
    const timer = window.setTimeout(() => setShow(true), 10);
    return () => window.clearTimeout(timer);
  }, [initialAlignmentMode, initialBlendMode, isOpen, sourcePaths]);

  useEffect(() => {
    if (!isOpen) return;
    const timer = window.setTimeout(() => closeButtonRef.current?.focus(), 0);
    return () => window.clearTimeout(timer);
  }, [isOpen]);

  useEffect(() => {
    if (isOpen) return;
    setShow(false);
    const timer = window.setTimeout(() => setIsMounted(false), 240);
    return () => window.clearTimeout(timer);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !isProcessing && !isSaving) {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isProcessing, isSaving, onClose]);

  const selectedAlignment = useMemo(
    () => ALIGNMENT_OPTIONS.find((option) => option.value === alignmentMode) || ALIGNMENT_OPTIONS[0],
    [alignmentMode],
  );

  const translateAlignment = (key: string) => t(`modals.imageStack.alignmentModes.${key}` as never) as string;

  const movePath = useCallback(
    (index: number, direction: -1 | 1) => {
      setOrderedPaths((current) => {
        const targetIndex = index + direction;
        if (targetIndex < 0 || targetIndex >= current.length) return current;
        const next = [...current];
        [next[index], next[targetIndex]] = [next[targetIndex], next[index]];
        return next;
      });
      setSavedPath(null);
      onChange();
    },
    [onChange],
  );

  const handleProcess = () => {
    if (isProcessing || orderedPaths.length < 2) return;
    setSavedPath(null);
    onProcess(orderedPaths, blendMode, alignmentMode);
  };

  const handleSave = async () => {
    if (isSaving || savedPath || !finalImageBase64) return;
    setIsSaving(true);
    try {
      const path = await onSave(blendMode);
      setSavedPath(path);
    } catch (saveError) {
      console.error('Failed to save image stack:', saveError);
    } finally {
      setIsSaving(false);
    }
  };

  if (!isMounted) return null;

  const canProcess = orderedPaths.length >= 2 && !isProcessing && !isSaving;

  return (
    <div
      aria-modal="true"
      aria-describedby="image-stack-description"
      aria-labelledby="image-stack-title"
      className={`fixed inset-0 z-50 flex items-center justify-center overscroll-contain bg-black/55 p-3 backdrop-blur-sm transition-opacity duration-200 motion-reduce:transition-none sm:p-6 ${
        show ? 'opacity-100' : 'opacity-0'
      }`}
      onMouseDown={(event) => {
        mouseDownTarget.current = event.target;
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget && mouseDownTarget.current === event.currentTarget && !isProcessing) {
          onClose();
        }
        mouseDownTarget.current = null;
      }}
      role="dialog"
    >
      <motion.div
        animate={
          shouldReduceMotion
            ? { opacity: show ? 1 : 0 }
            : show
              ? { opacity: 1, scale: 1, y: 0 }
              : { opacity: 0, scale: 0.97, y: 12 }
        }
        className="flex max-h-[min(900px,calc(100dvh-24px))] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border-color bg-surface shadow-2xl sm:max-h-[min(900px,calc(100dvh-48px))]"
        initial={false}
        onClick={(event) => event.stopPropagation()}
        onMouseDown={(event) => event.stopPropagation()}
        transition={{ duration: shouldReduceMotion ? 0 : 0.2, ease: [0.22, 1, 0.36, 1] }}
      >
        <header className="flex shrink-0 items-center justify-between gap-4 border-b border-border-color px-4 py-3 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent/15 text-accent">
              <Layers3 aria-hidden="true" size={20} />
            </div>
            <div className="min-w-0">
              <h2 className="truncate text-base font-semibold text-text-primary sm:text-lg" id="image-stack-title">
                {t('modals.imageStack.title')}
              </h2>
              <p className="truncate text-xs text-text-secondary" id="image-stack-description">
                {t('modals.imageStack.subtitle')}
              </p>
            </div>
          </div>
          <button
            aria-label={t('modals.imageStack.close')}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
            disabled={isProcessing || isSaving}
            ref={closeButtonRef}
            onClick={onClose}
            type="button"
          >
            <X aria-hidden="true" size={18} />
          </button>
        </header>

        <div className="grid min-h-0 flex-1 lg:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="min-h-0 overscroll-contain overflow-y-auto border-b border-border-color bg-bg-primary/35 p-4 lg:border-b-0 lg:border-r sm:p-5">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-text-primary">{t('modals.imageStack.sourceLayers')}</h3>
                <p className="mt-1 text-xs text-text-secondary">
                  {t('modals.imageStack.sourceCount', { count: orderedPaths.length })}
                </p>
              </div>
              <span className="rounded-full bg-card-active px-2 py-1 text-[11px] font-medium text-text-secondary">
                {t('modals.imageStack.orderHint')}
              </span>
            </div>

            <ol className="space-y-2" aria-label={t('modals.imageStack.sourceLayers')}>
              {orderedPaths.map((path, index) => (
                <li
                  className="flex items-center gap-2 rounded-xl border border-border-color bg-surface px-2 py-2"
                  key={`${path}-${index}`}
                >
                  <span className="w-5 shrink-0 text-center text-[11px] font-semibold text-text-secondary">
                    {index + 1}
                  </span>
                  <div className="h-11 w-14 shrink-0 overflow-hidden rounded-lg bg-bg-primary">
                    {thumbnails[path] ? (
                      <img
                        alt={getDisplayName(path)}
                        className="h-full w-full object-cover"
                        height={44}
                        loading="lazy"
                        src={thumbnails[path]}
                        width={56}
                      />
                    ) : (
                      <div className="h-full w-full bg-card-active" />
                    )}
                  </div>
                  <span className="min-w-0 flex-1 truncate text-xs text-text-primary" title={getDisplayName(path)}>
                    {getDisplayName(path)}
                  </span>
                  <div className="flex shrink-0 flex-col gap-0.5">
                    <button
                      aria-label={t('modals.imageStack.moveUp', { number: index + 1 })}
                      className="rounded p-1 text-text-secondary hover:bg-card-active hover:text-text-primary disabled:opacity-25"
                      disabled={index === 0 || isProcessing}
                      onClick={() => movePath(index, -1)}
                      type="button"
                    >
                      <ArrowUp aria-hidden="true" size={13} />
                    </button>
                    <button
                      aria-label={t('modals.imageStack.moveDown', { number: index + 1 })}
                      className="rounded p-1 text-text-secondary hover:bg-card-active hover:text-text-primary disabled:opacity-25"
                      disabled={index === orderedPaths.length - 1 || isProcessing}
                      onClick={() => movePath(index, 1)}
                      type="button"
                    >
                      <ArrowDown aria-hidden="true" size={13} />
                    </button>
                  </div>
                </li>
              ))}
            </ol>

            <div className="mt-5 rounded-xl border border-border-color bg-surface/70 p-3 text-xs leading-relaxed text-text-secondary">
              {t('modals.imageStack.sourceHint')}
            </div>
          </aside>

          <main className="min-h-0 overscroll-contain overflow-y-auto p-4 sm:p-5">
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_280px]">
              <section className="min-h-[300px] overflow-hidden rounded-2xl border border-border-color bg-[#101010] sm:min-h-[430px]">
                <div className="flex h-full min-h-[300px] items-center justify-center p-3 sm:min-h-[430px]">
                  {error ? (
                    <div className="max-w-md text-center">
                      <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-status-error/15 text-status-error">
                        <X aria-hidden="true" size={22} />
                      </div>
                      <h3 className="text-sm font-semibold text-text-primary" id="image-stack-error-title">
                        {t('modals.imageStack.failed')}
                      </h3>
                      <p className="mt-2 break-words rounded-lg bg-black/25 p-3 text-xs leading-relaxed text-text-secondary">
                        <span aria-live="assertive">{error}</span>
                      </p>
                    </div>
                  ) : finalImageBase64 ? (
                    <div className="relative flex h-full w-full items-center justify-center">
                      <img
                        alt={t('modals.imageStack.resultAlt')}
                        className="max-h-[520px] max-w-full object-contain"
                        height={520}
                        src={finalImageBase64}
                        width={900}
                      />
                      <div className="absolute left-3 top-3 rounded-full bg-black/55 px-2.5 py-1 text-[11px] text-white/85 backdrop-blur-sm">
                        {blendMode === 'focus'
                          ? t('modals.imageStack.focusStackResult')
                          : t('modals.imageStack.panoramaResult')}
                      </div>
                    </div>
                  ) : (
                    <div className="relative flex h-full w-full max-w-2xl items-center justify-center">
                      {orderedPaths.slice(0, 5).map((path, index) => (
                        <div
                          className="absolute h-[68%] w-[64%] overflow-hidden rounded-xl border border-white/20 bg-[#1c1c1c] shadow-2xl"
                          key={`${path}-preview-${index}`}
                          style={{
                            transform: `translate(${(index - 2) * 18}px, ${(2 - index) * 10}px) rotate(${(index - 2) * 1.8}deg)`,
                            zIndex: index,
                          }}
                        >
                          {thumbnails[path] ? (
                            <img
                              alt={getDisplayName(path)}
                              className="h-full w-full object-cover opacity-90"
                              height={360}
                              loading="lazy"
                              src={thumbnails[path]}
                              width={520}
                            />
                          ) : (
                            <div className="h-full w-full bg-card-active" />
                          )}
                        </div>
                      ))}
                      <div className="absolute bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-full bg-black/65 px-3 py-1.5 text-xs text-white/85 backdrop-blur-sm">
                        {t('modals.imageStack.readyToProcess')}
                      </div>
                    </div>
                  )}
                </div>
              </section>

              <section className="rounded-2xl border border-border-color bg-bg-primary/30 p-4">
                <div className="mb-3">
                  <h3 className="text-sm font-semibold text-text-primary">{t('modals.imageStack.workflow')}</h3>
                  <p className="mt-1 text-xs leading-relaxed text-text-secondary">
                    {t('modals.imageStack.workflowHint')}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-2" role="group" aria-label={t('modals.imageStack.workflow')}>
                  <button
                    aria-pressed={blendMode === 'focus'}
                    className={`rounded-xl border p-3 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                      blendMode === 'focus'
                        ? 'border-accent bg-accent/10 text-text-primary'
                        : 'border-border-color bg-surface text-text-secondary hover:bg-card-active'
                    }`}
                    disabled={isProcessing}
                    onClick={() => {
                      setBlendMode('focus');
                      setSavedPath(null);
                      onChange();
                    }}
                    type="button"
                  >
                    <Layers3 aria-hidden="true" size={18} />
                    <span className="mt-2 block text-xs font-semibold">{t('modals.imageStack.focusStack')}</span>
                    <span className="mt-1 block text-[11px] leading-relaxed opacity-75">
                      {t('modals.imageStack.focusStackDescription')}
                    </span>
                  </button>
                  <button
                    aria-pressed={blendMode === 'panorama'}
                    className={`rounded-xl border p-3 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                      blendMode === 'panorama'
                        ? 'border-accent bg-accent/10 text-text-primary'
                        : 'border-border-color bg-surface text-text-secondary hover:bg-card-active'
                    }`}
                    disabled={isProcessing}
                    onClick={() => {
                      setBlendMode('panorama');
                      setSavedPath(null);
                      onChange();
                    }}
                    type="button"
                  >
                    <Expand aria-hidden="true" size={18} />
                    <span className="mt-2 block text-xs font-semibold">{t('modals.imageStack.panorama')}</span>
                    <span className="mt-1 block text-[11px] leading-relaxed opacity-75">
                      {t('modals.imageStack.panoramaDescription')}
                    </span>
                  </button>
                </div>

                <div className="mt-5 border-t border-border-color pt-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-semibold text-text-primary">{t('modals.imageStack.alignment')}</h3>
                      <p className="mt-1 text-xs text-text-secondary">{t('modals.imageStack.alignmentHint')}</p>
                    </div>
                    <span className="rounded-full bg-card-active px-2 py-1 text-[11px] text-text-secondary">
                      {translateAlignment(selectedAlignment.labelKey)}
                    </span>
                  </div>
                  <div className="space-y-1.5" role="radiogroup" aria-label={t('modals.imageStack.alignment')}>
                    {ALIGNMENT_OPTIONS.map((option) => {
                      const Icon = option.icon;
                      const isSelected = option.value === alignmentMode;
                      return (
                        <button
                          aria-checked={isSelected}
                          className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                            isSelected
                              ? 'border-accent bg-accent/10 text-text-primary'
                              : 'border-transparent text-text-secondary hover:border-border-color hover:bg-card-active'
                          }`}
                          disabled={isProcessing}
                          key={option.value}
                          onClick={() => {
                            setAlignmentMode(option.value);
                            setSavedPath(null);
                            onChange();
                          }}
                          role="radio"
                          type="button"
                        >
                          <span
                            className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md ${
                              isSelected ? 'bg-accent text-white' : 'bg-card-active'
                            }`}
                          >
                            <Icon aria-hidden="true" size={15} />
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block text-xs font-medium">{translateAlignment(option.labelKey)}</span>
                            <span className="mt-0.5 block truncate text-[10px] opacity-70">
                              {translateAlignment(option.descriptionKey)}
                            </span>
                          </span>
                          {isSelected && <Check aria-hidden="true" className="shrink-0 text-accent" size={15} />}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </section>
            </div>

            <AnimatePresence initial={false} mode="wait">
              {isProcessing && (
                <motion.div
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 rounded-xl border border-border-color bg-bg-primary/40 p-3"
                  exit={{ opacity: 0, y: -4 }}
                  initial={{ opacity: 0, y: 4 }}
                  role="status"
                >
                  <TaskProgress
                    ariaLabel={t('modals.imageStack.processing')}
                    indeterminate
                    label={progressMessage || t('modals.imageStack.processing')}
                  />
                </motion.div>
              )}
            </AnimatePresence>
            {savedPath && (
              <div
                aria-live="polite"
                className="mt-4 flex items-center gap-2 rounded-xl border border-status-success/30 bg-status-success/10 px-3 py-2 text-xs text-status-success"
                role="status"
              >
                <Check aria-hidden="true" size={15} />
                <span className="min-w-0 flex-1 truncate">{t('modals.imageStack.savedSuccess')}</span>
                <button
                  className="font-medium underline underline-offset-2"
                  onClick={() => onOpenFile(savedPath)}
                  type="button"
                >
                  {t('modals.imageStack.openInEditor')}
                </button>
              </div>
            )}
          </main>
        </div>

        <footer className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-t border-border-color bg-bg-primary/25 px-4 py-3 sm:px-6">
          <p className="text-xs text-text-secondary">
            {t('modals.imageStack.selectedSummary', { count: orderedPaths.length })}
          </p>
          <div className="flex flex-wrap justify-end gap-2">
            <button
              className="rounded-lg px-3 py-2 text-xs text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary disabled:opacity-40"
              disabled={isProcessing || isSaving}
              onClick={onClose}
              type="button"
            >
              {t('modals.imageStack.cancel')}
            </button>
            <Button
              className="border border-border-color bg-surface text-text-primary hover:bg-card-active"
              disabled={!canProcess}
              onClick={handleProcess}
            >
              {isProcessing ? (
                <Loader2 aria-hidden="true" className="animate-spin" size={15} />
              ) : finalImageBase64 ? (
                <RefreshCw aria-hidden="true" size={15} />
              ) : (
                <Layers3 aria-hidden="true" size={15} />
              )}
              {finalImageBase64 ? t('modals.imageStack.realign') : t('modals.imageStack.start')}
            </Button>
            {finalImageBase64 && (
              <Button disabled={isSaving || isProcessing || Boolean(savedPath)} onClick={handleSave}>
                {isSaving ? (
                  <Loader2 aria-hidden="true" className="animate-spin" size={15} />
                ) : (
                  <Save aria-hidden="true" size={15} />
                )}
                {t('modals.imageStack.save')}
              </Button>
            )}
          </div>
        </footer>
      </motion.div>
    </div>
  );
}
