import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { invoke, isTauri } from '@tauri-apps/api/core';
import { open, save } from '@tauri-apps/plugin-dialog';
import {
  ArrowRight,
  ArrowLeft,
  Blend,
  Check,
  Download,
  FileImage,
  ImagePlus,
  Loader2,
  RefreshCw,
  Sparkles,
  X,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import Button from '../ui/Button';
import { Invokes } from '../ui/AppProperties';
import {
  analyzeStyleTransfer,
  applyStyleTransfer,
  summarizeStyleTransform,
  type ImageDataLike,
  type StyleTransferMode,
  type StyleTransferTransform,
} from '../../utils/styleTransfer';

const MAX_PREVIEW_EDGE = 1_600;
const DEFAULT_STRENGTH = 0.86;
const IMAGE_EXTENSIONS = [
  'jpg',
  'jpeg',
  'png',
  'gif',
  'bmp',
  'tiff',
  'tif',
  'webp',
  'jxl',
  'exr',
  'hdr',
  'tga',
  'ico',
  'dds',
  'qoi',
  'ff',
  'pnm',
  'pbm',
  'pgm',
  'ppm',
  'pam',
  'dng',
  'pro',
  'ari',
  'crw',
  'cr2',
  'cr3',
  'bay',
  'raw',
  'erf',
  'raf',
  '3fr',
  'fff',
  'iiq',
  'kdc',
  'k25',
  'dcs',
  'dcr',
  'mos',
  'rwl',
  'mef',
  'mrw',
  'nef',
  'nrw',
  'orf',
  'rw2',
  'pef',
  'ptx',
  'srw',
  'x3f',
  'arw',
  'srf',
  'sr2',
];

type SourceRole = 'reference' | 'target';

interface StyleImage {
  isPath: boolean;
  name: string;
  url: string;
  width: number;
  height: number;
  path?: string;
  image: HTMLImageElement;
}

interface StyleTransferModalProps {
  fullPage?: boolean;
  isOpen: boolean;
  onClose(): void;
}

interface SourceCardProps {
  image: StyleImage | null;
  isBusy: boolean;
  onChoose(): void;
  onDrop(event: React.DragEvent<HTMLDivElement>): void;
  onRemove(): void;
  role: SourceRole;
}

const readImageElement = (url: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The selected image could not be decoded.'));
    image.src = url;
  });

const getFileName = (path: string) => path.split(/[\\/]/).pop() || path;

const getOutputName = (name: string) => {
  const withoutExtension = name.replace(/\.[^.]+$/, '');
  return `${withoutExtension}_styled.jpg`;
};

const toUint8Array = (value: Uint8Array | ArrayBuffer) => (value instanceof Uint8Array ? value : new Uint8Array(value));

const makeImageData = (image: HTMLImageElement): ImageDataLike => {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const scale = Math.min(1, MAX_PREVIEW_EDGE / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas preview is unavailable in this environment.');
  context.drawImage(image, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
};

const toNativeImageData = (source: ImageDataLike): ImageData =>
  source instanceof ImageData ? source : new ImageData(new Uint8ClampedArray(source.data), source.width, source.height);

const createImageDataCopy = (source: ImageDataLike): ImageData => toNativeImageData(source);

function SourceCard({ image, isBusy, onChoose, onDrop, onRemove, role }: SourceCardProps) {
  const { t } = useTranslation();
  const isReference = role === 'reference';
  const title = isReference
    ? t('styleTransfer.reference', { defaultValue: 'Reference image' })
    : t('styleTransfer.target', { defaultValue: 'Target image' });
  const helper = isReference
    ? t('styleTransfer.referenceHint', { defaultValue: 'The image whose colour language you want to borrow.' })
    : t('styleTransfer.targetHint', { defaultValue: 'The image that keeps its composition and detail.' });

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onChoose();
    }
  };

  return (
    <div className="style-transfer-source-card" data-source-role={role}>
      <div className="style-transfer-source-heading">
        <span className="style-transfer-source-index">{isReference ? '01' : '02'}</span>
        <div className="min-w-0">
          <h3>{title}</h3>
          <p>{helper}</p>
        </div>
      </div>
      <div
        aria-label={image ? t('styleTransfer.replaceImage', { defaultValue: `Replace ${title}` }) : title}
        aria-busy={isBusy}
        className={`style-transfer-dropzone ${image ? 'has-image' : ''} ${isBusy ? 'is-busy' : ''}`}
        onClick={onChoose}
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
        onKeyDown={handleKeyDown}
        role="button"
        tabIndex={0}
      >
        {image ? (
          <>
            <img alt={image.name} src={image.url} />
            <div className="style-transfer-image-scrim" aria-hidden="true" />
            <div className="style-transfer-image-meta">
              <span className="style-transfer-image-name" title={image.name}>
                {image.name}
              </span>
              <span>
                {image.width.toLocaleString()} × {image.height.toLocaleString()}
              </span>
            </div>
            <button
              aria-label={t('styleTransfer.removeImage', { defaultValue: `Remove ${title}` })}
              className="style-transfer-remove"
              onClick={(event) => {
                event.stopPropagation();
                onRemove();
              }}
              type="button"
            >
              <X aria-hidden="true" size={14} />
            </button>
          </>
        ) : (
          <div className="style-transfer-empty-dropzone">
            {isBusy ? (
              <Loader2 aria-hidden="true" className="animate-spin" size={25} />
            ) : (
              <span className="style-transfer-upload-mark" aria-hidden="true">
                <ImagePlus size={22} />
              </span>
            )}
            <strong>
              {isBusy
                ? t('styleTransfer.loading', { defaultValue: 'Preparing preview…' })
                : t('styleTransfer.chooseImage', { defaultValue: 'Choose an image' })}
            </strong>
            <span>{t('styleTransfer.dropHint', { defaultValue: 'or drop it here · JPG, PNG, TIFF, RAW' })}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function StyleTransferModal({ fullPage = false, isOpen, onClose }: StyleTransferModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [reference, setReference] = useState<StyleImage | null>(null);
  const [target, setTarget] = useState<StyleImage | null>(null);
  const [mode, setMode] = useState<StyleTransferMode>('mood');
  const [strength, setStrength] = useState(DEFAULT_STRENGTH);
  const [comparePosition, setComparePosition] = useState(0.5);
  const [transform, setTransform] = useState<StyleTransferTransform | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [busyRole, setBusyRole] = useState<SourceRole | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [exportedPath, setExportedPath] = useState<string | null>(null);

  const inputRef = useRef<HTMLInputElement>(null);
  const pickerRoleRef = useRef<SourceRole>('reference');
  const originalCanvasRef = useRef<HTMLCanvasElement>(null);
  const resultCanvasRef = useRef<HTMLCanvasElement>(null);
  const originalDataRef = useRef<ImageDataLike | null>(null);
  const resultDataRef = useRef<ImageDataLike | null>(null);
  const objectUrlsRef = useRef<Set<string>>(new Set());

  const tr = useCallback(
    (key: string, fallback: string, options?: Record<string, unknown>) =>
      t(key, { defaultValue: fallback, ...(options || {}) }),
    [t],
  );

  const revokeObjectUrls = useCallback(() => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current.clear();
  }, []);

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = window.setTimeout(() => setShow(true), 10);
      return () => window.clearTimeout(timer);
    }

    setShow(false);
    const timer = window.setTimeout(() => {
      setIsMounted(false);
      setReference(null);
      setTarget(null);
      setTransform(null);
      setError(null);
      setExportedPath(null);
      setStrength(DEFAULT_STRENGTH);
      setMode('mood');
      setIsProcessing(false);
      originalDataRef.current = null;
      resultDataRef.current = null;
      revokeObjectUrls();
    }, 220);
    return () => window.clearTimeout(timer);
  }, [isOpen, revokeObjectUrls]);

  useEffect(() => {
    return revokeObjectUrls;
  }, [revokeObjectUrls]);

  useEffect(() => {
    if (!isOpen) return;
    const handleWindowKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !isExporting) {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', handleWindowKeyDown);
    return () => window.removeEventListener('keydown', handleWindowKeyDown);
  }, [isExporting, isOpen, onClose]);

  const rememberUrl = (url: string) => {
    objectUrlsRef.current.add(url);
    return url;
  };

  const releaseUrl = (url: string) => {
    URL.revokeObjectURL(url);
    objectUrlsRef.current.delete(url);
  };

  const loadFile = useCallback(async (file: File): Promise<StyleImage> => {
    const url = rememberUrl(URL.createObjectURL(file));
    try {
      const image = await readImageElement(url);
      return {
        isPath: false,
        name: file.name,
        url,
        width: image.naturalWidth || image.width,
        height: image.naturalHeight || image.height,
        image,
      };
    } catch (error) {
      releaseUrl(url);
      throw error;
    }
  }, []);

  const loadPath = useCallback(async (path: string): Promise<StyleImage> => {
    const previewResult = await invoke<Uint8Array | ArrayBuffer>(Invokes.GenerateStyleTransferPreview, { path });
    const bytes = toUint8Array(previewResult);
    const jpegBytes = new Uint8Array(bytes);
    // Keep the typed-array view's byte offset/length intact. Tauri may return
    // a view into a larger IPC buffer, and handing Blob the backing buffer
    // directly could prepend or append unrelated bytes to the JPEG stream.
    const url = rememberUrl(URL.createObjectURL(new Blob([jpegBytes], { type: 'image/jpeg' })));
    try {
      const image = await readImageElement(url);
      return {
        isPath: true,
        name: getFileName(path),
        path,
        url,
        // The native preview is intentionally bounded to keep the modal
        // responsive. The full-resolution source is loaded again only when
        // the user exports, so these dimensions describe the working preview.
        width: image.naturalWidth || image.width,
        height: image.naturalHeight || image.height,
        image,
      };
    } catch (error) {
      releaseUrl(url);
      throw error;
    }
  }, []);

  const setSource = useCallback(
    (role: SourceRole, image: StyleImage) => {
      setError(null);
      setExportedPath(null);
      const previous = role === 'reference' ? reference : target;
      if (previous && previous.url !== image.url && objectUrlsRef.current.has(previous.url)) {
        URL.revokeObjectURL(previous.url);
        objectUrlsRef.current.delete(previous.url);
      }
      if (role === 'reference') setReference(image);
      else setTarget(image);
    },
    [reference, target],
  );

  const chooseSource = useCallback(
    async (role: SourceRole) => {
      setError(null);
      pickerRoleRef.current = role;

      if (!isTauri()) {
        inputRef.current?.click();
        return;
      }

      try {
        setBusyRole(role);
        const selected = await open({
          multiple: false,
          title:
            role === 'reference'
              ? tr('styleTransfer.chooseReference', 'Choose reference image')
              : tr('styleTransfer.chooseTarget', 'Choose target image'),
          filters: [{ name: tr('styleTransfer.imageFiles', 'Image files'), extensions: IMAGE_EXTENSIONS }],
        });
        if (typeof selected === 'string') {
          const image = await loadPath(selected);
          setSource(role, image);
        }
      } catch (selectionError) {
        console.error('Failed to load style transfer image:', selectionError);
        setError(tr('styleTransfer.loadError', 'Could not load that image. Try another file.'));
      } finally {
        setBusyRole(null);
      }
    },
    [loadPath, setSource, tr],
  );

  const handleInputChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;
      try {
        const role = pickerRoleRef.current;
        setBusyRole(role);
        setSource(role, await loadFile(file));
      } catch (selectionError) {
        console.error('Failed to decode style transfer image:', selectionError);
        setError(tr('styleTransfer.loadError', 'Could not load that image. Try another file.'));
      } finally {
        setBusyRole(null);
      }
    },
    [loadFile, setSource, tr],
  );

  const handleDrop = useCallback(
    async (role: SourceRole, event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      const file = event.dataTransfer.files?.[0];
      if (!file) return;
      try {
        setBusyRole(role);
        setSource(role, await loadFile(file));
      } catch (dropError) {
        console.error('Failed to decode dropped style transfer image:', dropError);
        setError(tr('styleTransfer.loadError', 'Could not load that image. Try another file.'));
      } finally {
        setBusyRole(null);
      }
    },
    [loadFile, setSource, tr],
  );

  const removeSource = useCallback(
    (role: SourceRole) => {
      const previous = role === 'reference' ? reference : target;
      if (previous && objectUrlsRef.current.has(previous.url)) {
        URL.revokeObjectURL(previous.url);
        objectUrlsRef.current.delete(previous.url);
      }
      if (role === 'reference') setReference(null);
      else setTarget(null);
      setTransform(null);
      setError(null);
      setExportedPath(null);
    },
    [reference, target],
  );

  useEffect(() => {
    if (!reference || !target) {
      setIsProcessing(false);
      setTransform(null);
      originalDataRef.current = null;
      resultDataRef.current = null;
      return;
    }

    let cancelled = false;
    setIsProcessing(true);
    setError(null);

    const frame = window.requestAnimationFrame(() => {
      try {
        const referenceData = makeImageData(reference.image);
        const targetData = makeImageData(target.image);
        const nextTransform = analyzeStyleTransfer(referenceData, targetData, mode);
        const resultData = createImageDataCopy(targetData);
        applyStyleTransfer(resultData, nextTransform, strength);
        if (cancelled) return;

        originalDataRef.current = targetData;
        resultDataRef.current = resultData;
        setTransform(nextTransform);
      } catch (processingError) {
        console.error('Failed to create style transfer preview:', processingError);
        if (!cancelled) setError(tr('styleTransfer.previewError', 'Preview could not be created for this pair.'));
      } finally {
        if (!cancelled) setIsProcessing(false);
      }
    });

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [mode, reference, strength, target, tr]);

  useEffect(() => {
    const originalData = originalDataRef.current;
    const resultData = resultDataRef.current;
    const originalCanvas = originalCanvasRef.current;
    const resultCanvas = resultCanvasRef.current;
    if (!originalData || !resultData || !originalCanvas || !resultCanvas) return;

    originalCanvas.width = originalData.width;
    originalCanvas.height = originalData.height;
    resultCanvas.width = resultData.width;
    resultCanvas.height = resultData.height;
    originalCanvas.getContext('2d')?.putImageData(toNativeImageData(originalData), 0, 0);
    resultCanvas.getContext('2d')?.putImageData(toNativeImageData(resultData), 0, 0);
  }, [transform]);

  const summary = useMemo(() => (transform ? summarizeStyleTransform(transform) : null), [transform]);
  const canExport = Boolean(reference && target && transform && !isProcessing && !isExporting);
  const outputName = target ? getOutputName(target.name) : 'styled-image.jpg';

  const handleExport = useCallback(async () => {
    if (!target || !reference || !transform || isExporting) return;
    setIsExporting(true);
    setError(null);
    setExportedPath(null);

    try {
      if (isTauri() && reference.isPath && target.isPath && reference.path && target.path) {
        const selectedOutputPath = await save({
          defaultPath: getOutputName(target.name),
          filters: [{ name: tr('styleTransfer.jpegFile', 'JPEG image'), extensions: ['jpg', 'jpeg'] }],
          title: tr('styleTransfer.exportTitle', 'Export styled image'),
        });
        if (!selectedOutputPath) return;

        const savedPath = await invoke<string>(Invokes.ExportStyleTransfer, {
          referencePath: reference.path,
          targetPath: target.path,
          outputPath: selectedOutputPath,
          strength,
          mode,
        });
        setExportedPath(savedPath);
      } else {
        const resultData = resultDataRef.current;
        if (!resultData) throw new Error('No rendered result is available.');
        const canvas = document.createElement('canvas');
        canvas.width = resultData.width;
        canvas.height = resultData.height;
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Canvas export is unavailable.');
        context.putImageData(toNativeImageData(resultData), 0, 0);
        const blob = await new Promise<Blob>((resolve, reject) => {
          canvas.toBlob(
            (value) => (value ? resolve(value) : reject(new Error('Could not encode the image.'))),
            'image/jpeg',
            0.94,
          );
        });
        const downloadUrl = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = downloadUrl;
        anchor.download = outputName;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1_000);
        setExportedPath(outputName);
      }
    } catch (exportError) {
      console.error('Failed to export style transfer result:', exportError);
      setError(tr('styleTransfer.exportError', 'Export failed. Check the destination and try again.'));
    } finally {
      setIsExporting(false);
    }
  }, [isExporting, mode, outputName, reference, strength, target, tr, transform]);

  if (!isMounted) return null;

  return (
    <div
      aria-labelledby="style-transfer-title"
      aria-describedby="style-transfer-description"
      aria-modal={fullPage ? undefined : true}
      className={
        fullPage
          ? `style-transfer-page ${show ? 'is-visible' : ''}`
          : `app-modal-backdrop style-transfer-backdrop ${show ? 'opacity-100' : 'opacity-0'}`
      }
      onClick={fullPage ? undefined : () => !isExporting && onClose()}
      role={fullPage ? 'main' : 'dialog'}
    >
      <section
        className={
          fullPage
            ? `style-transfer-page-surface ${show ? 'is-visible' : ''}`
            : `app-modal-surface app-modal-surface--structured style-transfer-modal ${show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.985] opacity-0'}`
        }
        onClick={(event) => event.stopPropagation()}
      >
        <header className="style-transfer-header">
          <div className="style-transfer-header-leading">
            {fullPage && (
              <Button
                aria-label={tr('styleTransfer.backToLibrary', 'Back to library')}
                className="style-transfer-back"
                disabled={isExporting}
                onClick={onClose}
                type="button"
                variant="ghost"
              >
                <ArrowLeft aria-hidden="true" size={15} />
                {tr('styleTransfer.backToLibrary', 'Back to library')}
              </Button>
            )}
            <div className="style-transfer-heading">
              <span className="style-transfer-eyebrow">
                <Sparkles aria-hidden="true" size={13} />
                {tr('styleTransfer.eyebrow', 'STYLE LAB')}
              </span>
              <h2 id="style-transfer-title">{tr('styleTransfer.title', 'Transfer a look')}</h2>
              <p id="style-transfer-description">
                {tr(
                  'styleTransfer.description',
                  'Borrow the colour language of one image while keeping the subject and detail of another.',
                )}
              </p>
            </div>
          </div>
          {!fullPage && (
            <button
              aria-label={tr('styleTransfer.close', 'Close style transfer')}
              className="style-transfer-close"
              disabled={isExporting}
              onClick={onClose}
              type="button"
            >
              <X aria-hidden="true" size={18} />
            </button>
          )}
        </header>

        <div className="style-transfer-body">
          <input ref={inputRef} accept="image/*" className="hidden" onChange={handleInputChange} type="file" />

          <div className="style-transfer-source-row">
            <SourceCard
              image={reference}
              isBusy={busyRole === 'reference'}
              onChoose={() => void chooseSource('reference')}
              onDrop={(event) => void handleDrop('reference', event)}
              onRemove={() => removeSource('reference')}
              role="reference"
            />
            <div className="style-transfer-flow" aria-hidden="true">
              <span />
              <ArrowRight size={17} />
              <span />
            </div>
            <SourceCard
              image={target}
              isBusy={busyRole === 'target'}
              onChoose={() => void chooseSource('target')}
              onDrop={(event) => void handleDrop('target', event)}
              onRemove={() => removeSource('target')}
              role="target"
            />
          </div>

          <div className="style-transfer-workbench">
            <div className="style-transfer-preview-panel">
              <div className="style-transfer-panel-heading">
                <div>
                  <span className="style-transfer-panel-kicker">
                    {tr('styleTransfer.previewKicker', 'LIVE PREVIEW')}
                  </span>
                  <h3>{tr('styleTransfer.previewTitle', 'See the mood move')}</h3>
                </div>
                {target && (
                  <span className="style-transfer-preview-size">
                    {target.width.toLocaleString()} × {target.height.toLocaleString()}
                  </span>
                )}
              </div>

              <div
                className={`style-transfer-preview-stage ${target ? 'has-target' : ''}`}
                style={target ? { aspectRatio: `${target.width} / ${target.height}` } : undefined}
              >
                {target ? (
                  <>
                    <canvas
                      aria-label={tr('styleTransfer.styledPreview', 'Styled target preview')}
                      ref={resultCanvasRef}
                    />
                    <div
                      className="style-transfer-original-layer"
                      style={{ clipPath: `inset(0 ${Math.max(0, (1 - comparePosition) * 100)}% 0 0)` }}
                    >
                      <canvas
                        aria-label={tr('styleTransfer.originalPreview', 'Original target preview')}
                        ref={originalCanvasRef}
                      />
                    </div>
                    <div className="style-transfer-compare-line" style={{ left: `${comparePosition * 100}%` }}>
                      <span />
                    </div>
                    <span className="style-transfer-preview-label style-transfer-preview-label--left">
                      {tr('styleTransfer.original', 'Original')}
                    </span>
                    <span className="style-transfer-preview-label style-transfer-preview-label--right">
                      {tr('styleTransfer.styled', 'Styled')}
                    </span>
                    {isProcessing && (
                      <div className="style-transfer-processing-indicator">
                        <Loader2 aria-hidden="true" className="animate-spin" size={15} />
                        {tr('styleTransfer.updating', 'Updating preview…')}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="style-transfer-preview-empty">
                    <Blend aria-hidden="true" size={30} strokeWidth={1.4} />
                    <strong>{tr('styleTransfer.previewEmptyTitle', 'Your result will appear here')}</strong>
                    <span>
                      {tr('styleTransfer.previewEmptyHint', 'Add a reference and target image to start comparing.')}
                    </span>
                  </div>
                )}
              </div>

              {target && (
                <label className="style-transfer-compare-control">
                  <span>{tr('styleTransfer.compare', 'Compare')}</span>
                  <input
                    aria-label={tr('styleTransfer.compare', 'Compare')}
                    max="100"
                    min="0"
                    onChange={(event) => setComparePosition(Number(event.target.value) / 100)}
                    style={{ '--range-progress': `${comparePosition * 100}%` } as React.CSSProperties}
                    type="range"
                    value={Math.round(comparePosition * 100)}
                  />
                  <span className="tabular-nums">{Math.round(comparePosition * 100)}%</span>
                </label>
              )}
            </div>

            <aside
              className="style-transfer-controls"
              aria-label={tr('styleTransfer.controls', 'Style transfer controls')}
            >
              <div className="style-transfer-control-section">
                <div className="style-transfer-control-label">
                  <span>{tr('styleTransfer.method', 'Matching method')}</span>
                  <span className="style-transfer-control-note">{tr('styleTransfer.localOnly', 'Runs locally')}</span>
                </div>
                <div
                  className="style-transfer-mode-switch"
                  role="group"
                  aria-label={tr('styleTransfer.method', 'Matching method')}
                >
                  <button
                    aria-pressed={mode === 'mood'}
                    className={mode === 'mood' ? 'is-active' : ''}
                    onClick={() => setMode('mood')}
                    type="button"
                  >
                    {tr('styleTransfer.mood', 'Mood')}
                    <small>{tr('styleTransfer.moodHint', 'Hue + tone')}</small>
                  </button>
                  <button
                    aria-pressed={mode === 'distribution'}
                    className={mode === 'distribution' ? 'is-active' : ''}
                    onClick={() => setMode('distribution')}
                    type="button"
                  >
                    {tr('styleTransfer.distribution', 'Distribution')}
                    <small>{tr('styleTransfer.distributionHint', 'RGB balance')}</small>
                  </button>
                </div>
              </div>

              <div className="style-transfer-control-section">
                <div className="style-transfer-control-label">
                  <span>{tr('styleTransfer.strength', 'Transfer strength')}</span>
                  <span className="tabular-nums">{Math.round(strength * 100)}%</span>
                </div>
                <input
                  aria-label={tr('styleTransfer.strength', 'Transfer strength')}
                  className="style-transfer-range"
                  max="100"
                  min="0"
                  onChange={(event) => setStrength(Number(event.target.value) / 100)}
                  style={{ '--range-progress': `${strength * 100}%` } as React.CSSProperties}
                  type="range"
                  value={Math.round(strength * 100)}
                />
                <div className="style-transfer-range-labels">
                  <span>{tr('styleTransfer.subtle', 'Subtle')}</span>
                  <span>{tr('styleTransfer.full', 'Full')}</span>
                </div>
              </div>

              <div className="style-transfer-metrics" aria-live="polite">
                <div className="style-transfer-metrics-heading">
                  <span>{tr('styleTransfer.readout', 'Transfer readout')}</span>
                  {transform && <Check aria-hidden="true" size={14} />}
                </div>
                <div className="style-transfer-metric-grid">
                  <div>
                    <span>{tr('styleTransfer.hue', 'Hue')}</span>
                    <strong>{summary ? `${summary.hueDegrees > 0 ? '+' : ''}${summary.hueDegrees}°` : '—'}</strong>
                  </div>
                  <div>
                    <span>{tr('styleTransfer.saturation', 'Saturation')}</span>
                    <strong>
                      {summary ? `${summary.saturationPercent > 0 ? '+' : ''}${summary.saturationPercent}%` : '—'}
                    </strong>
                  </div>
                  <div>
                    <span>{tr('styleTransfer.brightness', 'Brightness')}</span>
                    <strong>
                      {summary ? `${summary.brightnessPercent > 0 ? '+' : ''}${summary.brightnessPercent}%` : '—'}
                    </strong>
                  </div>
                  <div>
                    <span>{tr('styleTransfer.contrast', 'Contrast')}</span>
                    <strong>
                      {summary ? `${summary.contrastPercent > 0 ? '+' : ''}${summary.contrastPercent}%` : '—'}
                    </strong>
                  </div>
                </div>
              </div>

              <div className="style-transfer-local-note">
                <FileImage aria-hidden="true" size={15} />
                <span>{tr('styleTransfer.privacy', 'Images stay on this device. No upload or account required.')}</span>
              </div>
            </aside>
          </div>
        </div>

        <footer className="style-transfer-footer">
          <div className="style-transfer-status" aria-live="polite">
            {error ? (
              <span className="style-transfer-status-error">{error}</span>
            ) : exportedPath ? (
              <span className="style-transfer-status-success">
                <Check aria-hidden="true" size={14} />
                {tr('styleTransfer.exported', 'Exported')} · {exportedPath}
              </span>
            ) : (
              <span>
                {tr(
                  'styleTransfer.footerHint',
                  'A preview is generated from a compact copy; your originals remain untouched.',
                )}
              </span>
            )}
          </div>
          <div className="style-transfer-actions">
            <Button
              className="style-transfer-reset"
              disabled={isExporting}
              onClick={() => {
                setMode('mood');
                setStrength(DEFAULT_STRENGTH);
                setComparePosition(0.5);
                setError(null);
                setExportedPath(null);
              }}
              type="button"
              variant="ghost"
            >
              <RefreshCw aria-hidden="true" size={14} />
              {tr('styleTransfer.reset', 'Reset')}
            </Button>
            <Button disabled={!canExport} onClick={() => void handleExport()} type="button">
              {isExporting ? (
                <Loader2 aria-hidden="true" className="animate-spin" size={15} />
              ) : (
                <Download aria-hidden="true" size={15} />
              )}
              {isExporting
                ? tr('styleTransfer.exporting', 'Exporting…')
                : tr('styleTransfer.export', 'Export styled image')}
            </Button>
          </div>
        </footer>
      </section>
    </div>
  );
}
