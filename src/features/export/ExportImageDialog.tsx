import { useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, Link2, Loader2, Save, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import Button from '../../components/ui/Button';
import Switch from '../../components/ui/Switch';
import ZoomableImagePreview from '../../components/preview/ZoomableImagePreview';
import {
  EXPORT_DIALOG_FORMATS,
  buildExportMetadataEntries,
  clampExportDimension,
  clampExportPercent,
  createInitialExportDialogSettings,
  dimensionsFromPercent,
  estimateExportFileSize,
  heightFromWidth,
  widthFromHeight,
} from './exportDialog';
import type { ExportDialogFormat, ExportDialogSettings, ExportDialogSource, ExportMetadataMode } from './exportDialog';

interface ExportImageDialogProps {
  initialFormat?: ExportDialogFormat;
  isOpen: boolean;
  metadata?: Record<string, unknown> | null;
  onClose(): void;
  onEstimateSize?(settings: ExportDialogSettings): Promise<number | null>;
  onExport(settings: ExportDialogSettings): Promise<string | null>;
  onExported?(path: string): void;
  source: ExportDialogSource | null;
}

const fieldClassName =
  'h-9 w-full rounded-md border border-border-color bg-bg-primary/55 px-2.5 text-xs text-text-primary outline-none transition-colors placeholder:text-text-secondary/55 focus:border-accent focus:ring-1 focus:ring-accent/30 disabled:opacity-50';

const sectionClassName = 'rounded-lg border border-border-color bg-bg-primary/26 p-3';

const formatMetadataKey = (key: string): string =>
  key
    .replace(/[_-]+/g, ' ')
    .replace(/([a-z\d])([A-Z])/g, '$1 $2')
    .replace(/([A-Z])([A-Z][a-z])/g, '$1 $2')
    .replace(/([A-Za-z])(\d)/g, '$1 $2')
    .replace(/(\d)([A-Za-z])/g, '$1 $2');

function NumberField({
  label,
  onChange,
  suffix,
  value,
}: {
  label: string;
  onChange(value: number): void;
  suffix: string;
  value: number;
}) {
  const [draftValue, setDraftValue] = useState(String(value));

  useEffect(() => {
    setDraftValue(String(value));
  }, [value]);

  const commitValue = () => {
    const parsed = Number(draftValue);
    if (draftValue.trim() && Number.isFinite(parsed)) {
      onChange(parsed);
      return;
    }
    setDraftValue(String(value));
  };

  return (
    <label className="grid grid-cols-[4rem_minmax(0,1fr)_2.5rem] items-center gap-2 text-xs text-text-secondary">
      <span>{label}</span>
      <input
        className={fieldClassName}
        inputMode="numeric"
        min={1}
        onBlur={commitValue}
        onChange={(event) => {
          const nextValue = event.target.value;
          setDraftValue(nextValue);
          if (!nextValue.trim()) return;
          const parsed = Number(nextValue);
          if (Number.isFinite(parsed)) onChange(parsed);
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter') event.currentTarget.blur();
          if (event.key === 'Escape') {
            setDraftValue(String(value));
            event.currentTarget.blur();
          }
        }}
        type="number"
        value={draftValue}
      />
      <span className="text-[11px]">{suffix}</span>
    </label>
  );
}

function MetadataModeButton({ checked, label, onClick }: { checked: boolean; label: string; onClick(): void }) {
  return (
    <button
      aria-checked={checked}
      className={`flex h-9 items-center justify-between rounded-md border px-2.5 text-left text-xs transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${
        checked
          ? 'border-accent bg-accent/12 text-text-primary'
          : 'border-border-color bg-bg-primary/35 text-text-secondary hover:bg-card-active hover:text-text-primary'
      }`}
      onClick={onClick}
      role="radio"
      type="button"
    >
      {label}
      {checked && <Check aria-hidden="true" className="text-accent" size={14} />}
    </button>
  );
}

export default function ExportImageDialog({
  initialFormat = 'jpeg',
  isOpen,
  metadata,
  onClose,
  onEstimateSize,
  onExport,
  onExported,
  source,
}: ExportImageDialogProps) {
  const { i18n, t } = useTranslation();
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const [settings, setSettings] = useState<ExportDialogSettings | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isMetadataExpanded, setIsMetadataExpanded] = useState(false);
  const [refinedSizeEstimate, setRefinedSizeEstimate] = useState<{
    bytes: number;
    signature: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || !source) return;
    setSettings(createInitialExportDialogSettings(source, initialFormat, metadata));
    setError(null);
    setIsExporting(false);
    setIsMetadataExpanded(false);
    setRefinedSizeEstimate(null);
  }, [initialFormat, isOpen, metadata, source]);

  const estimateSignature = settings
    ? [
        source?.fileName,
        settings.format,
        settings.resizeWidth,
        settings.resizeHeight,
        settings.jpegQuality,
        settings.embedColorProfile,
        settings.metadataMode,
        settings.stripGps,
        settings.artist,
        settings.contact,
        settings.copyright,
        settings.description,
        ...Object.values(settings.metadataEditedFields),
      ].join('|')
    : '';

  useEffect(() => {
    if (!isOpen || !settings || !onEstimateSize) return;
    let isCancelled = false;
    setRefinedSizeEstimate(null);
    const timer = window.setTimeout(() => {
      void onEstimateSize(settings)
        .then((bytes) => {
          if (!isCancelled && bytes !== null && Number.isFinite(bytes) && bytes > 0) {
            setRefinedSizeEstimate({ bytes, signature: estimateSignature });
          }
        })
        .catch(() => {
          // The immediate pixel-based estimate remains available if native estimation fails.
        });
    }, 350);
    return () => {
      isCancelled = true;
      window.clearTimeout(timer);
    };
  }, [estimateSignature, isOpen, metadata, onEstimateSize, source]);

  useEffect(() => {
    if (!isOpen) return;
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusTimer = window.setTimeout(() => closeButtonRef.current?.focus(), 0);
    return () => {
      window.clearTimeout(focusTimer);
      if (previouslyFocused?.isConnected) previouslyFocused.focus();
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !isExporting) {
        event.preventDefault();
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== 'Tab') return;
      const focusable = Array.from(
        dialogRef.current?.querySelectorAll<HTMLElement>(
          'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ) ?? [],
      );
      if (focusable.length === 0) return;
      const currentIndex = focusable.indexOf(document.activeElement as HTMLElement);
      const nextIndex = event.shiftKey
        ? currentIndex <= 0
          ? focusable.length - 1
          : currentIndex - 1
        : currentIndex === focusable.length - 1
          ? 0
          : currentIndex + 1;
      event.preventDefault();
      focusable[nextIndex].focus();
    };
    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, [isExporting, isOpen, onClose]);

  const sourcePixels = useMemo(() => {
    if (!source) return 0;
    return source.width * source.height;
  }, [source]);

  if (!isOpen || !source || !settings) return null;

  const updateSettings = (patch: Partial<ExportDialogSettings>) =>
    setSettings((current) => (current ? { ...current, ...patch } : current));

  const updateMetadataField = (field: 'artist' | 'contact' | 'copyright' | 'description', value: string) =>
    setSettings((current) =>
      current
        ? {
            ...current,
            [field]: value,
            metadataEditedFields: { ...current.metadataEditedFields, [field]: true },
          }
        : current,
    );

  const updateWidth = (value: number) => {
    const resizeWidth = clampExportDimension(value);
    const resizeHeight = heightFromWidth(source.width, source.height, resizeWidth);
    updateSettings({
      resizeHeight,
      resizePercent: clampExportPercent((resizeWidth / source.width) * 100),
      resizeWidth,
    });
  };

  const updateHeight = (value: number) => {
    const resizeHeight = clampExportDimension(value);
    const resizeWidth = widthFromHeight(source.width, source.height, resizeHeight);
    updateSettings({
      resizeHeight,
      resizePercent: clampExportPercent((resizeHeight / source.height) * 100),
      resizeWidth,
    });
  };

  const updatePercent = (value: number) => {
    const resizePercent = clampExportPercent(value);
    const dimensions = dimensionsFromPercent(source.width, source.height, resizePercent);
    updateSettings({ resizePercent, resizeWidth: dimensions.width, resizeHeight: dimensions.height });
  };

  const handleExport = async () => {
    if (isExporting) return;
    setIsExporting(true);
    setError(null);
    try {
      const path = await onExport(settings);
      if (!path) {
        setIsExporting(false);
        return;
      }
      onExported?.(path);
      onClose();
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : String(exportError));
      setIsExporting(false);
    }
  };

  const metadataModeOptions: Array<{ id: ExportMetadataMode; label: string }> = [
    { id: 'none', label: t('export.exportDialog.metadataNone') },
    { id: 'copyright', label: t('export.exportDialog.metadataCopyright') },
    { id: 'all', label: t('export.exportDialog.metadataAll') },
  ];
  const copyrightFields: Array<{
    field: 'artist' | 'contact' | 'copyright' | 'description';
    label: string;
    maxLength: number;
  }> = [
    { field: 'artist', label: t('export.exportDialog.artist'), maxLength: 512 },
    { field: 'copyright', label: t('export.exportDialog.copyright'), maxLength: 512 },
    { field: 'contact', label: t('export.exportDialog.contact'), maxLength: 512 },
  ];
  if (settings.metadataMode === 'all') {
    copyrightFields.push({ field: 'description', label: t('export.exportDialog.description'), maxLength: 2_048 });
  }
  const outputPixels = settings.resizeWidth * settings.resizeHeight;
  const pixelRatio = sourcePixels > 0 ? outputPixels / sourcePixels : 1;
  const metadataEntries = buildExportMetadataEntries(metadata, settings);
  const estimatedFileSizes = Object.fromEntries(
    EXPORT_DIALOG_FORMATS.map((format) => [
      format.id,
      estimateExportFileSize(format.id, settings.resizeWidth, settings.resizeHeight, settings.jpegQuality),
    ]),
  ) as Record<ExportDialogFormat, number>;
  if (refinedSizeEstimate?.signature === estimateSignature) {
    estimatedFileSizes[settings.format] = refinedSizeEstimate.bytes;
  }
  const formatEstimatedBytes = (bytes: number): string => {
    const unitKeys = [
      'export.bytes.bytes',
      'export.bytes.kb',
      'export.bytes.mb',
      'export.bytes.gb',
      'export.bytes.tb',
    ] as const;
    const unitIndex = bytes > 0 ? Math.min(unitKeys.length - 1, Math.floor(Math.log(bytes) / Math.log(1_024))) : 0;
    const value = bytes / Math.pow(1_024, unitIndex);
    const maximumFractionDigits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
    return `${new Intl.NumberFormat(i18n.resolvedLanguage, { maximumFractionDigits }).format(value)} ${t(
      unitKeys[unitIndex],
    )}`;
  };

  return (
    <div
      aria-busy={isExporting}
      aria-labelledby="export-image-dialog-title"
      aria-modal="true"
      className="fixed inset-0 z-[80] flex items-center justify-center bg-black/72 p-2 backdrop-blur-[2px] sm:p-4"
      role="dialog"
    >
      <div
        className="flex h-[min(900px,calc(100dvh-1rem))] w-[min(1500px,calc(100vw-1rem))] flex-col overflow-hidden rounded-xl border border-border-color bg-surface shadow-2xl sm:h-[min(900px,calc(100dvh-2rem))] sm:w-[min(1500px,calc(100vw-2rem))]"
        ref={dialogRef}
      >
        <header className="flex h-13 shrink-0 items-center justify-between border-b border-border-color px-4">
          <div className="min-w-0">
            <h2 className="truncate text-sm font-semibold text-text-primary" id="export-image-dialog-title">
              {t('export.exportDialog.title')}
            </h2>
            <p className="truncate text-[11px] text-text-secondary">{source.fileName}</p>
          </div>
          <button
            aria-label={t('export.exportDialog.close')}
            className="flex h-8 w-8 items-center justify-center rounded-md text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40"
            disabled={isExporting}
            onClick={onClose}
            ref={closeButtonRef}
            type="button"
          >
            <X aria-hidden="true" size={17} />
          </button>
        </header>

        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_24rem]">
          <section className="relative min-h-[18rem] overflow-hidden bg-bg-primary/80 lg:min-h-0">
            <ZoomableImagePreview
              alt={source.fileName}
              className="h-full w-full"
              detailSrc={source.detailPreviewSrc}
              interactionSrc={source.previewSrc}
              labels={{
                fit: t('export.exportDialog.previewFit'),
                toolbar: t('export.exportDialog.previewToolbar'),
                zoomIn: t('export.exportDialog.previewZoomIn'),
                zoomOut: t('export.exportDialog.previewZoomOut'),
              }}
              sourceSize={{ width: settings.resizeWidth, height: settings.resizeHeight }}
            />
            <div className="pointer-events-none absolute left-3 top-3 rounded-md border border-white/10 bg-black/68 px-2 py-1 text-[11px] text-white/78 shadow-sm">
              {source.width} × {source.height} → {settings.resizeWidth} × {settings.resizeHeight}
            </div>
          </section>

          <aside className="min-h-0 overflow-y-auto border-l border-border-color bg-surface p-3 custom-scrollbar">
            <div className="space-y-3">
              <section className={sectionClassName}>
                <h3 className="mb-2 text-xs font-semibold text-text-primary">{t('export.sections.fileSettings')}</h3>
                <div
                  aria-label={t('export.sections.fileSettings')}
                  className="grid grid-cols-3 gap-1.5"
                  role="radiogroup"
                >
                  {EXPORT_DIALOG_FORMATS.map((format) => (
                    <button
                      aria-checked={settings.format === format.id}
                      aria-label={`${format.label}, ${t('export.status.estimatedSize', {
                        size: formatEstimatedBytes(estimatedFileSizes[format.id]),
                      })}`}
                      className={`flex min-h-12 flex-col items-center justify-center rounded-md border px-1 py-1.5 text-xs font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${
                        settings.format === format.id
                          ? 'border-accent bg-accent/13 text-text-primary'
                          : 'border-border-color bg-bg-primary/45 text-text-secondary hover:bg-card-active hover:text-text-primary'
                      }`}
                      key={format.id}
                      onClick={() => updateSettings({ format: format.id })}
                      role="radio"
                      type="button"
                    >
                      <span>{format.label}</span>
                      <span
                        className={`mt-0.5 text-[10px] font-normal tabular-nums ${
                          settings.format === format.id ? 'text-accent' : 'text-text-secondary/80'
                        }`}
                      >
                        ≈ {formatEstimatedBytes(estimatedFileSizes[format.id])}
                      </span>
                    </button>
                  ))}
                </div>
                {settings.format === 'jpeg' && (
                  <label className="mt-3 block text-xs text-text-secondary">
                    <span className="mb-1.5 flex items-center justify-between">
                      {t('export.file.quality')}
                      <strong className="font-medium text-text-primary">{settings.jpegQuality}</strong>
                    </span>
                    <input
                      aria-label={t('export.file.quality')}
                      className="w-full accent-[var(--color-accent)]"
                      max={100}
                      min={1}
                      onChange={(event) => updateSettings({ jpegQuality: Number(event.target.value) })}
                      type="range"
                      value={settings.jpegQuality}
                    />
                  </label>
                )}
              </section>

              <section className={sectionClassName}>
                <div className="mb-2 flex items-center justify-between gap-2">
                  <h3 className="text-xs font-semibold text-text-primary">{t('export.sections.imageSizing')}</h3>
                  <span className="text-[10px] text-text-secondary">
                    {pixelRatio >= 1 ? `${pixelRatio.toFixed(2)}×` : `${Math.round(pixelRatio * 100)}%`}{' '}
                    {t('export.exportDialog.pixels')}
                  </span>
                </div>
                <div className="relative space-y-2">
                  <NumberField
                    label={t('export.exportDialog.width')}
                    onChange={updateWidth}
                    suffix="px"
                    value={settings.resizeWidth}
                  />
                  <div className="pointer-events-none absolute right-[2.8rem] top-[1.75rem] flex h-5 w-5 items-center justify-center rounded-full border border-border-color bg-surface text-text-secondary">
                    <Link2 aria-hidden="true" size={11} />
                  </div>
                  <NumberField
                    label={t('export.exportDialog.height')}
                    onChange={updateHeight}
                    suffix="px"
                    value={settings.resizeHeight}
                  />
                  <NumberField
                    label={t('export.exportDialog.scale')}
                    onChange={updatePercent}
                    suffix="%"
                    value={settings.resizePercent}
                  />
                  <label className="grid grid-cols-[4rem_minmax(0,1fr)] items-center gap-2 text-xs text-text-secondary">
                    <span>{t('export.exportDialog.resample')}</span>
                    <span className="relative">
                      <select className={`${fieldClassName} appearance-none pr-8`} disabled value="lanczos3">
                        <option value="lanczos3">{t('export.exportDialog.resampleLanczos')}</option>
                      </select>
                      <ChevronDown
                        aria-hidden="true"
                        className="pointer-events-none absolute right-2.5 top-2.5"
                        size={14}
                      />
                    </span>
                  </label>
                </div>
              </section>

              <section className={sectionClassName}>
                <h3 className="mb-2 text-xs font-semibold text-text-primary">{t('export.sections.metadata')}</h3>
                <div aria-label={t('export.sections.metadata')} className="grid gap-1.5" role="radiogroup">
                  {metadataModeOptions.map((option) => (
                    <MetadataModeButton
                      checked={settings.metadataMode === option.id}
                      key={option.id}
                      label={option.label}
                      onClick={() => updateSettings({ metadataMode: option.id })}
                    />
                  ))}
                </div>
                {settings.metadataMode === 'all' && (
                  <Switch
                    checked={settings.stripGps}
                    className="mt-3"
                    id="export-dialog-strip-gps"
                    label={t('export.metadata.removeGps')}
                    onChange={(stripGps) => updateSettings({ stripGps })}
                  />
                )}
                <div className="mt-3 border-t border-border-color/75 pt-1.5">
                  <button
                    aria-controls="export-dialog-metadata-details"
                    aria-expanded={isMetadataExpanded}
                    className="flex min-h-8 w-full items-center justify-between gap-3 rounded-md px-1.5 text-left text-xs text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
                    onClick={() => setIsMetadataExpanded((expanded) => !expanded)}
                    type="button"
                  >
                    <span className="font-medium">{t('editor.metadata.extendedExif.title')}</span>
                    <span className="flex shrink-0 items-center gap-1.5 tabular-nums">
                      <span>{metadataEntries.length}</span>
                      <ChevronDown
                        aria-hidden="true"
                        className={`transition-transform duration-150 motion-reduce:transition-none ${
                          isMetadataExpanded ? 'rotate-180' : ''
                        }`}
                        size={14}
                      />
                    </span>
                  </button>
                  {isMetadataExpanded && (
                    <div className="pt-1.5" id="export-dialog-metadata-details">
                      {metadataEntries.length > 0 ? (
                        <dl className="max-h-52 overflow-y-auto rounded-md bg-bg-primary/32 px-2 custom-scrollbar">
                          {metadataEntries.map((entry) => (
                            <div
                              className="grid grid-cols-[minmax(5.75rem,0.8fr)_minmax(0,1.2fr)] gap-3 border-b border-border-color/55 py-2 last:border-b-0"
                              key={entry.key}
                            >
                              <dt className="break-words text-[10px] font-medium leading-4 text-text-secondary">
                                {formatMetadataKey(entry.key)}
                              </dt>
                              <dd className="min-w-0 whitespace-pre-wrap break-words text-[10px] leading-4 text-text-primary [overflow-wrap:anywhere]">
                                {entry.value}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      ) : (
                        <p className="rounded-md bg-bg-primary/32 px-2.5 py-2 text-[10px] text-text-secondary">
                          {t('export.exportDialog.metadataNone')}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </section>

              {settings.metadataMode !== 'none' && (
                <section className={sectionClassName}>
                  <h3 className="mb-2 text-xs font-semibold text-text-primary">
                    {t('export.exportDialog.copyrightTitle')}
                  </h3>
                  <div className="grid gap-2">
                    {copyrightFields.map(({ field, label, maxLength }) => (
                      <label className="grid gap-1 text-[11px] text-text-secondary" key={field}>
                        <span>{label}</span>
                        {field === 'description' ? (
                          <textarea
                            className={`${fieldClassName} h-16 resize-y py-2`}
                            maxLength={maxLength}
                            onChange={(event) => updateMetadataField(field, event.target.value)}
                            value={settings[field]}
                          />
                        ) : (
                          <input
                            className={fieldClassName}
                            maxLength={maxLength}
                            onChange={(event) => updateMetadataField(field, event.target.value)}
                            type="text"
                            value={settings[field]}
                          />
                        )}
                      </label>
                    ))}
                  </div>
                </section>
              )}

              <section className={sectionClassName}>
                <h3 className="mb-2 text-xs font-semibold text-text-primary">{t('export.exportDialog.colorSpace')}</h3>
                <div className="mb-2 flex h-9 items-center justify-between rounded-md border border-border-color bg-bg-primary/35 px-2.5 text-xs">
                  <span className="text-text-secondary">{t('export.exportDialog.convertTo')}</span>
                  <strong className="font-medium text-text-primary">{t('export.exportDialog.srgb')}</strong>
                </div>
                <Switch
                  checked={settings.embedColorProfile}
                  id="export-dialog-embed-profile"
                  label={t('export.exportDialog.embedProfile')}
                  onChange={(embedColorProfile) => updateSettings({ embedColorProfile })}
                />
              </section>
            </div>
          </aside>
        </div>

        <footer className="flex min-h-14 shrink-0 items-center justify-between gap-3 border-t border-border-color px-4 py-2">
          <div className="min-w-0">
            {error ? (
              <p className="truncate text-xs text-status-error" role="alert">
                {error}
              </p>
            ) : (
              <p className="text-[11px] text-text-secondary">
                {settings.format === 'tiff' || settings.format === 'png'
                  ? t('export.exportDialog.bitDepth16')
                  : t('export.exportDialog.bitDepth8')}
              </p>
            )}
          </div>
          <div className="flex shrink-0 gap-2">
            <Button
              className="border border-border-color bg-surface text-text-primary hover:bg-card-active"
              disabled={isExporting}
              onClick={onClose}
            >
              {t('export.exportDialog.cancel')}
            </Button>
            <Button disabled={isExporting} onClick={handleExport}>
              {isExporting ? (
                <Loader2 aria-hidden="true" className="animate-spin" size={15} />
              ) : (
                <Save aria-hidden="true" size={15} />
              )}
              {isExporting ? t('export.status.exporting') : t('export.exportDialog.export')}
            </Button>
          </div>
        </footer>
      </div>
    </div>
  );
}
