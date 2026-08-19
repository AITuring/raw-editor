import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties } from 'react';
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import type { CollisionDetection, DragEndEvent } from '@dnd-kit/core';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  Check,
  Cylinder,
  Expand,
  Globe2,
  GripVertical,
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
import ImageStackResultPreview from './ImageStackResultPreview';

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

const closestLayerCenter: CollisionDetection = (args) =>
  closestCenter({
    ...args,
    droppableContainers: args.droppableContainers.filter((container) => container.id !== args.active.id),
  });

interface SourceLayerItemProps {
  count: number;
  disabled: boolean;
  dragLabel: string;
  index: number;
  moveDownLabel: string;
  moveUpLabel: string;
  onMove(index: number, direction: -1 | 1): void;
  path: string;
  thumbnail?: string;
}

function SourceLayerItem({
  count,
  disabled,
  dragLabel,
  index,
  moveDownLabel,
  moveUpLabel,
  onMove,
  path,
  thumbnail,
}: SourceLayerItemProps) {
  const {
    attributes,
    isDragging,
    listeners,
    setNodeRef: setDraggableNodeRef,
    transform,
  } = useDraggable({
    disabled,
    id: path,
  });
  const { isOver, setNodeRef: setDroppableNodeRef } = useDroppable({
    disabled,
    id: path,
  });

  const setCombinedRef = useCallback(
    (node: HTMLLIElement | null) => {
      setDraggableNodeRef(node);
      setDroppableNodeRef(node);
    },
    [setDraggableNodeRef, setDroppableNodeRef],
  );

  const style: CSSProperties = {
    opacity: isDragging ? 0.42 : 1,
    transform: transform ? `translate3d(${transform.x}px, ${transform.y}px, 0)` : undefined,
    zIndex: isDragging ? 20 : undefined,
  };

  return (
    <li
      className={`relative flex min-h-14 items-center gap-1.5 rounded-lg border bg-bg-primary/30 px-1.5 py-1.5 transition-[border-color,background-color,box-shadow,opacity] ${
        isOver && !isDragging
          ? 'border-accent bg-accent/8 shadow-[inset_3px_0_0_var(--color-accent)]'
          : 'border-border-color hover:bg-card-active/65'
      }`}
      ref={setCombinedRef}
      style={style}
    >
      <div
        {...listeners}
        className={`flex min-w-0 flex-1 touch-none items-center gap-1.5 ${
          disabled ? 'cursor-default' : 'cursor-grab active:cursor-grabbing'
        }`}
      >
        <button
          {...attributes}
          aria-label={dragLabel}
          className="flex h-10 w-7 shrink-0 items-center justify-center rounded-md text-text-secondary transition-colors hover:bg-surface hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:cursor-default disabled:opacity-35"
          disabled={disabled}
          type="button"
        >
          <GripVertical aria-hidden="true" size={15} />
        </button>
        <span className="w-4 shrink-0 text-center text-[10px] font-semibold tabular-nums text-text-secondary">
          {index + 1}
        </span>
        <div className="h-10 w-12 shrink-0 overflow-hidden rounded-md bg-bg-primary ring-1 ring-inset ring-border-color/60">
          {thumbnail ? (
            <img
              alt={getDisplayName(path)}
              className="h-full w-full object-cover"
              draggable={false}
              height={40}
              loading="lazy"
              src={thumbnail}
              width={48}
            />
          ) : (
            <div className="h-full w-full bg-card-active" />
          )}
        </div>
        <span className="min-w-0 flex-1 truncate text-xs text-text-primary" title={getDisplayName(path)}>
          {getDisplayName(path)}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-0.5">
        <button
          aria-label={moveUpLabel}
          className="flex h-8 w-7 items-center justify-center rounded-md text-text-secondary transition-colors hover:bg-surface hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-20"
          disabled={index === 0 || disabled}
          onClick={() => onMove(index, -1)}
          type="button"
        >
          <ArrowUp aria-hidden="true" size={13} />
        </button>
        <button
          aria-label={moveDownLabel}
          className="flex h-8 w-7 items-center justify-center rounded-md text-text-secondary transition-colors hover:bg-surface hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-20"
          disabled={index === count - 1 || disabled}
          onClick={() => onMove(index, 1)}
          type="button"
        >
          <ArrowDown aria-hidden="true" size={13} />
        </button>
      </div>
    </li>
  );
}

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
  const [isPreviewFocused, setIsPreviewFocused] = useState(false);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const shouldReduceMotion = useReducedMotion();
  const layerSensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 6 },
    }),
    useSensor(KeyboardSensor),
  );

  useEffect(() => {
    if (!isOpen) return;
    setOrderedPaths(sourcePaths);
    setBlendMode(initialBlendMode);
    setAlignmentMode(initialAlignmentMode);
    setSavedPath(null);
    setIsPreviewFocused(false);
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
    if (!finalImageBase64) setIsPreviewFocused(false);
  }, [finalImageBase64]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && isPreviewFocused) {
        event.preventDefault();
        setIsPreviewFocused(false);
      } else if (event.key === 'Escape' && !isSaving) {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isPreviewFocused, isSaving, onClose]);

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

  const handleLayerDragEnd = useCallback(
    ({ active, over }: DragEndEvent) => {
      if (!over || active.id === over.id || isProcessing) return;
      const activeIndex = orderedPaths.indexOf(String(active.id));
      const overIndex = orderedPaths.indexOf(String(over.id));
      if (activeIndex < 0 || overIndex < 0) return;

      const next = [...orderedPaths];
      const [movedPath] = next.splice(activeIndex, 1);
      next.splice(overIndex, 0, movedPath);
      setOrderedPaths(next);
      setSavedPath(null);
      onChange();
    },
    [isProcessing, onChange, orderedPaths],
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
  const SelectedAlignmentIcon = selectedAlignment.icon;

  return (
    <div
      aria-modal="true"
      aria-describedby="image-stack-description"
      aria-labelledby="image-stack-title"
      className={`fixed inset-0 z-50 overflow-hidden bg-surface transition-opacity duration-150 motion-reduce:transition-none ${
        show ? 'opacity-100' : 'opacity-0'
      }`}
      role="dialog"
    >
      <motion.div
        animate={shouldReduceMotion ? { opacity: show ? 1 : 0 } : { opacity: show ? 1 : 0, y: show ? 0 : 6 }}
        className="flex h-dvh w-full flex-col overflow-hidden bg-surface"
        initial={false}
        transition={{ duration: shouldReduceMotion ? 0 : 0.16, ease: [0.22, 1, 0.36, 1] }}
      >
        <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border-color bg-surface px-3 sm:px-4">
          <div className="flex min-w-0 items-center gap-3">
            <button
              aria-label={t('modals.imageStack.back')}
              className="flex h-9 shrink-0 items-center gap-1.5 rounded-lg px-2 text-xs font-medium text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
              disabled={isSaving}
              onClick={onClose}
              ref={closeButtonRef}
              type="button"
            >
              <ArrowLeft aria-hidden="true" size={17} />
              <span className="hidden sm:inline">{t('modals.imageStack.back')}</span>
            </button>
            <div aria-hidden="true" className="h-6 w-px shrink-0 bg-border-color" />
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent/15 text-accent">
              <Layers3 aria-hidden="true" size={18} />
            </div>
            <div className="min-w-0">
              <h2 className="truncate text-sm font-semibold text-text-primary" id="image-stack-title">
                {t('modals.imageStack.title')}
              </h2>
              <p className="truncate text-[11px] text-text-secondary" id="image-stack-description">
                {t('modals.imageStack.subtitle')}
              </p>
            </div>
          </div>
          <button
            aria-label={t('modals.imageStack.close')}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-40"
            disabled={isSaving}
            onClick={onClose}
            type="button"
          >
            <X aria-hidden="true" size={18} />
          </button>
        </header>

        <div
          className={`grid min-h-0 flex-1 ${
            isPreviewFocused ? 'grid-cols-1' : 'grid-cols-[minmax(0,1fr)_clamp(304px,24vw,368px)]'
          }`}
        >
          <main className="relative min-h-0 overflow-hidden bg-[#101010]">
            <section className="relative h-full min-h-0 overflow-hidden">
              <div className="flex h-full min-h-0 items-center justify-center">
                {error ? (
                  <div className="max-w-md p-6 text-center">
                    <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-status-error/15 text-status-error">
                      <X aria-hidden="true" size={22} />
                    </div>
                    <h3 className="text-sm font-semibold text-white" id="image-stack-error-title">
                      {t('modals.imageStack.failed')}
                    </h3>
                    <p className="mt-2 break-words rounded-lg bg-black/35 p-3 text-xs leading-relaxed text-white/65">
                      <span aria-live="assertive">{error}</span>
                    </p>
                  </div>
                ) : finalImageBase64 ? (
                  <ImageStackResultPreview
                    alignmentLabel={translateAlignment(selectedAlignment.labelKey)}
                    alt={t('modals.imageStack.resultAlt')}
                    isFocused={isPreviewFocused}
                    modeLabel={
                      blendMode === 'focus'
                        ? t('modals.imageStack.focusStackResult')
                        : t('modals.imageStack.panoramaResult')
                    }
                    onFocusedChange={setIsPreviewFocused}
                    src={finalImageBase64}
                  />
                ) : (
                  <div className="relative flex h-full w-full items-center justify-center overflow-hidden">
                    {orderedPaths.slice(0, 5).map((path, index) => (
                      <div
                        className="absolute h-[70%] max-h-[720px] w-[68%] max-w-[1040px] overflow-hidden rounded-lg border border-white/20 bg-[#1c1c1c] shadow-2xl"
                        key={`${path}-preview-${index}`}
                        style={{
                          transform: `translate(${(index - 2) * 18}px, ${(2 - index) * 10}px) rotate(${(index - 2) * 1.5}deg)`,
                          zIndex: index,
                        }}
                      >
                        {thumbnails[path] ? (
                          <img
                            alt={getDisplayName(path)}
                            className="h-full w-full object-cover opacity-90"
                            draggable={false}
                            height={720}
                            loading="lazy"
                            src={thumbnails[path]}
                            width={1040}
                          />
                        ) : (
                          <div className="h-full w-full bg-card-active" />
                        )}
                      </div>
                    ))}
                    <div className="absolute bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-md bg-black/70 px-3 py-1.5 text-xs text-white/85 backdrop-blur-sm">
                      {t('modals.imageStack.readyToProcess')}
                    </div>
                  </div>
                )}
              </div>
            </section>

            <AnimatePresence initial={false} mode="wait">
              {isProcessing && (
                <motion.div
                  animate={{ opacity: 1, y: 0 }}
                  className="absolute left-1/2 top-4 z-30 w-[min(520px,calc(100%-32px))] -translate-x-1/2 rounded-xl border border-white/15 bg-black/75 p-3 text-white shadow-2xl backdrop-blur-md"
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
                className="absolute left-1/2 top-4 z-30 flex w-[min(520px,calc(100%-32px))] -translate-x-1/2 items-center gap-2 rounded-xl border border-status-success/35 bg-black/80 px-3 py-2 text-xs text-status-success shadow-2xl backdrop-blur-md"
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

          {!isPreviewFocused && (
            <aside className="min-h-0 overscroll-contain overflow-y-auto border-l border-border-color bg-surface">
              <section className="p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <h3 className="text-sm font-semibold text-text-primary">{t('modals.imageStack.sourceLayers')}</h3>
                    <p className="mt-0.5 text-xs text-text-secondary">
                      {t('modals.imageStack.sourceCount', { count: orderedPaths.length })}
                    </p>
                  </div>
                  <span className="rounded-md bg-card-active px-2 py-1 text-[10px] font-medium text-text-secondary">
                    {t('modals.imageStack.orderHint')}
                  </span>
                </div>

                <DndContext
                  collisionDetection={closestLayerCenter}
                  id="image-stack-layers-dnd"
                  onDragEnd={handleLayerDragEnd}
                  sensors={layerSensors}
                >
                  <ol className="space-y-1.5" aria-label={t('modals.imageStack.sourceLayers')}>
                    {orderedPaths.map((path, index) => (
                      <SourceLayerItem
                        count={orderedPaths.length}
                        disabled={isProcessing || isSaving}
                        dragLabel={t('modals.imageStack.dragLayer', { number: index + 1 })}
                        index={index}
                        key={path}
                        moveDownLabel={t('modals.imageStack.moveDown', { number: index + 1 })}
                        moveUpLabel={t('modals.imageStack.moveUp', { number: index + 1 })}
                        onMove={movePath}
                        path={path}
                        thumbnail={thumbnails[path]}
                      />
                    ))}
                  </ol>
                </DndContext>

                <p className="mt-3 border-t border-border-color pt-3 text-[11px] leading-relaxed text-text-secondary">
                  {t('modals.imageStack.sourceHint')}
                </p>
              </section>

              <section className="border-t border-border-color p-4">
                <h3 className="text-sm font-semibold text-text-primary">{t('modals.imageStack.workflow')}</h3>
                <div
                  className="mt-2 grid grid-cols-2 rounded-lg border border-border-color bg-bg-primary/45 p-1"
                  role="group"
                  aria-label={t('modals.imageStack.workflow')}
                >
                  <button
                    aria-pressed={blendMode === 'focus'}
                    className={`flex h-10 items-center justify-center gap-2 rounded-md px-2 text-xs font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                      blendMode === 'focus'
                        ? 'bg-surface text-text-primary shadow-sm'
                        : 'text-text-secondary hover:bg-card-active hover:text-text-primary'
                    }`}
                    disabled={isProcessing}
                    onClick={() => {
                      setBlendMode('focus');
                      setSavedPath(null);
                      onChange();
                    }}
                    type="button"
                  >
                    <Layers3 aria-hidden="true" size={16} />
                    <span>{t('modals.imageStack.focusStack')}</span>
                    {blendMode === 'focus' && <Check aria-hidden="true" className="text-accent" size={14} />}
                  </button>
                  <button
                    aria-pressed={blendMode === 'panorama'}
                    className={`flex h-10 items-center justify-center gap-2 rounded-md px-2 text-xs font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                      blendMode === 'panorama'
                        ? 'bg-surface text-text-primary shadow-sm'
                        : 'text-text-secondary hover:bg-card-active hover:text-text-primary'
                    }`}
                    disabled={isProcessing}
                    onClick={() => {
                      setBlendMode('panorama');
                      setSavedPath(null);
                      onChange();
                    }}
                    type="button"
                  >
                    <Expand aria-hidden="true" size={16} />
                    <span>{t('modals.imageStack.panorama')}</span>
                    {blendMode === 'panorama' && <Check aria-hidden="true" className="text-accent" size={14} />}
                  </button>
                </div>
                <p className="mt-2 text-[11px] leading-relaxed text-text-secondary">
                  {blendMode === 'focus'
                    ? t('modals.imageStack.focusStackDescription')
                    : t('modals.imageStack.panoramaDescription')}
                </p>
              </section>

              <section className="border-t border-border-color p-4">
                <div className="flex items-center justify-between gap-3">
                  <h3
                    className="text-sm font-semibold text-text-primary"
                    data-tooltip={t('modals.imageStack.alignmentHint')}
                  >
                    {t('modals.imageStack.alignment')}
                  </h3>
                  <span className="flex items-center gap-1.5 rounded-md bg-card-active px-2 py-1 text-[10px] font-medium text-text-secondary">
                    <SelectedAlignmentIcon aria-hidden="true" size={12} />
                    {translateAlignment(selectedAlignment.labelKey)}
                  </span>
                </div>
                <div
                  className="mt-2 grid grid-cols-2 gap-1.5"
                  role="radiogroup"
                  aria-label={t('modals.imageStack.alignment')}
                >
                  {ALIGNMENT_OPTIONS.map((option) => {
                    const Icon = option.icon;
                    const isSelected = option.value === alignmentMode;
                    return (
                      <button
                        aria-checked={isSelected}
                        className={`flex h-11 min-w-0 items-center gap-2 rounded-lg border px-2 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                          isSelected
                            ? 'border-accent bg-accent/10 text-text-primary'
                            : 'border-border-color/60 text-text-secondary hover:border-border-color hover:bg-card-active hover:text-text-primary'
                        }`}
                        data-tooltip={translateAlignment(option.descriptionKey)}
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
                        <Icon aria-hidden="true" className="shrink-0" size={15} />
                        <span className="min-w-0 flex-1 truncate text-[11px] font-medium">
                          {translateAlignment(option.labelKey)}
                        </span>
                        {isSelected && <Check aria-hidden="true" className="shrink-0 text-accent" size={13} />}
                      </button>
                    );
                  })}
                </div>
                <p className="mt-2 flex items-start gap-2 text-[11px] leading-relaxed text-text-secondary">
                  <SelectedAlignmentIcon aria-hidden="true" className="mt-0.5 shrink-0" size={13} />
                  <span>{translateAlignment(selectedAlignment.descriptionKey)}</span>
                </p>
              </section>
            </aside>
          )}
        </div>

        <footer className="flex h-14 shrink-0 items-center justify-between gap-3 border-t border-border-color bg-surface px-4">
          <p className="text-xs text-text-secondary">
            {t('modals.imageStack.selectedSummary', { count: orderedPaths.length })}
          </p>
          <div className="flex justify-end gap-2">
            <button
              className="rounded-lg px-3 py-2 text-xs text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary disabled:opacity-40"
              disabled={isSaving}
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
