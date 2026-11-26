import { useMemo, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Crosshair,
  Download,
  FileSearch,
  FolderOpen,
  Lock,
  Redo2,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  SquareDashedMousePointer,
  Trash2,
  Undo2,
  Upload,
  XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { open } from '@tauri-apps/plugin-dialog';
import { Invokes } from '../ui/AppProperties';
import type {
  CriterionFilter,
  MarkerFilter,
  StackReportInspection,
  StackReportInspectionMarker,
} from '../../utils/stackReportInspection';
import {
  buildDiagnosticSummaryExport,
  filterInspectionCriteria,
  filterInspectionMarkers,
  parseStackReportInspection,
} from '../../utils/stackReportInspection';
import type {
  RustExportBindingAudit,
  SourceLockDocument,
  SourceLockPolicy,
  WorldBox,
} from '../../utils/sourceLockSidecar';
import {
  buildVerifyExportBundleRequest,
  canRedo,
  canUndo,
  commitAddManualReviewRegion,
  commitAddSourceLock,
  commitRedo,
  commitRemoveManualReviewRegion,
  commitRemoveSourceLock,
  commitUndo,
  computeRegionPixelArea,
  computeWorldRegion,
  createSourceLockDocument,
  importAndReconcileSidecar,
  serializeSidecarFiles,
  sha256Hex,
  summarizeSidecarForExport,
  worldBoxToPolygon,
} from '../../utils/sourceLockSidecar';

interface ImageStackInspectionPanelProps {
  draftWorldBox?: WorldBox | null;
  error: string | null;
  exportedTiffPath?: string | null;
  inspection: StackReportInspection | null;
  isDrawingRegion?: boolean;
  loadedReportFileName?: string | null;
  selectedMarkerId: string | null;
  sourceLockDoc: SourceLockDocument | null;
  stackReportPath: string | null;
  onClearDraftWorldBox?(): void;
  onFocusMarker(marker: StackReportInspectionMarker): void;
  onFocusWorldBox?(worldBox: WorldBox): void;
  onLoadInspection(inspection: StackReportInspection, fileName: string): void;
  onReloadCurrentReport?(): void;
  onSourceLockDocChange(doc: SourceLockDocument | null): void;
  onToggleDrawRegion?(): void;
}

const formatGiB = (bytes: number | null): string => {
  if (bytes === null || bytes <= 0) return '—';
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
};

const formatSeconds = (seconds: number | null): string => {
  if (seconds === null || seconds < 0) return '—';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const rem = Math.round(seconds % 60);
  return `${mins}m ${rem}s`;
};

const fileBaseName = (filePath: string): string => filePath.split(/[\\/]/).pop() ?? filePath;

const downloadTextFile = (fileName: string, content: string, mimeType: string) => {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

export const buildInitialSidecarFromInspection = (inspection: StackReportInspection): SourceLockDocument => {
  const bindings = inspection.sources
    .filter((source) => !source.excluded && source.path.trim().length > 0)
    .map((source, index) => ({
      source_id: index + 1,
      path: source.path,
      sha256: source.sha256Before || sha256Hex(source.path),
      station_id: 0,
      capture_group_id: 0,
      source_to_world_sha256: sha256Hex(`${inspection.runId}:${source.path}`),
    }));
  const manifestSha = sha256Hex(bindings.map((binding) => `${binding.path}:${binding.sha256}`).join('\n'));
  return createSourceLockDocument({
    runId: inspection.runId,
    inputManifestSha256: manifestSha,
    coordinateFrame: {
      id: 'world-v1',
      canvas: [inspection.canvas?.width ?? 1, inspection.canvas?.height ?? 1],
      origin: [inspection.origin.x, inspection.origin.y],
      units: 'native_output_pixels',
    },
    sourceBindings: bindings,
  });
};

const criterionBadgeClass = (verdict: string): string => {
  if (verdict === 'pass') return 'bg-status-success/15 text-status-success';
  if (verdict === 'fail') return 'bg-status-error/15 text-status-error';
  if (verdict === 'insufficient_evidence') return 'bg-status-warning/15 text-status-warning';
  return 'bg-card-active text-text-secondary';
};

export default function ImageStackInspectionPanel({
  draftWorldBox = null,
  error,
  exportedTiffPath = null,
  inspection,
  isDrawingRegion = false,
  loadedReportFileName = null,
  selectedMarkerId,
  sourceLockDoc,
  stackReportPath,
  onClearDraftWorldBox,
  onFocusMarker,
  onFocusWorldBox,
  onLoadInspection,
  onReloadCurrentReport,
  onSourceLockDocChange,
  onToggleDrawRegion,
}: ImageStackInspectionPanelProps) {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const sidecarInputRef = useRef<HTMLInputElement | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadedFileName, setLoadedFileName] = useState<string | null>(null);
  const [criterionFilter, setCriterionFilter] = useState<CriterionFilter>('all');
  const [markerFilter, setMarkerFilter] = useState<MarkerFilter>('all');
  const [showSources, setShowSources] = useState(false);
  const [lockError, setLockError] = useState<string | null>(null);
  const [sidecarNotice, setSidecarNotice] = useState<string | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState<number>(1);
  const [selectedAllowedSourceIds, setSelectedAllowedSourceIds] = useState<number[]>([1]);
  const [lockPolicy, setLockPolicy] = useState<SourceLockPolicy>('single_source');
  const [reviewReason, setReviewReason] = useState('alignment_review');
  const [confirmStaleExport, setConfirmStaleExport] = useState(false);
  const [auditTiffPath, setAuditTiffPath] = useState<string>('');
  const [auditSidecarPath, setAuditSidecarPath] = useState<string>('');
  const [auditEventsPath, setAuditEventsPath] = useState<string>('');
  const [auditRecordedTiffSha, setAuditRecordedTiffSha] = useState<string>('');
  const [auditRecordedSidecarSha, setAuditRecordedSidecarSha] = useState<string>('');
  const [auditCheckCurrentRun, setAuditCheckCurrentRun] = useState<boolean>(true);
  const [isAuditingBundle, setIsAuditingBundle] = useState<boolean>(false);
  const [bundleAudit, setBundleAudit] = useState<RustExportBindingAudit | null>(null);
  const [bundleAuditError, setBundleAuditError] = useState<string | null>(null);

  const baselineDoc = useMemo(() => (inspection ? buildInitialSidecarFromInspection(inspection) : null), [inspection]);

  const activeDoc = sourceLockDoc ?? baselineDoc;

  const selectedMarker = useMemo(
    () => inspection?.markers.find((marker) => marker.id === selectedMarkerId) ?? inspection?.markers[0] ?? null,
    [inspection, selectedMarkerId],
  );

  const sourceNameById = useMemo(() => {
    const map = new Map<number, string>();
    for (const binding of activeDoc?.source_bindings ?? []) {
      map.set(binding.source_id, fileBaseName(binding.path));
    }
    return map;
  }, [activeDoc]);

  const targetRegionInfo = useMemo(() => {
    if (draftWorldBox) {
      const polygon = worldBoxToPolygon(draftWorldBox);
      return {
        kind: 'drawn' as const,
        box: draftWorldBox,
        polygon,
        areaPx: computeRegionPixelArea(polygon),
      };
    }
    if (selectedMarker) {
      const polygon = worldBoxToPolygon(selectedMarker.worldBox);
      return {
        kind: 'marker' as const,
        box: selectedMarker.worldBox,
        polygon,
        areaPx: computeRegionPixelArea(polygon),
      };
    }
    const defaultBox: WorldBox = {
      x: inspection?.origin.x ?? 0,
      y: inspection?.origin.y ?? 0,
      width: Math.min(512, inspection?.canvas?.width ?? 512),
      height: Math.min(512, inspection?.canvas?.height ?? 512),
    };
    const polygon = worldBoxToPolygon(defaultBox);
    return {
      kind: 'default' as const,
      box: defaultBox,
      polygon,
      areaPx: computeRegionPixelArea(polygon),
    };
  }, [draftWorldBox, inspection, selectedMarker]);

  const exportSummary = useMemo(() => (activeDoc ? summarizeSidecarForExport(activeDoc) : null), [activeDoc]);
  const visibleCriteria = useMemo(
    () => (inspection ? filterInspectionCriteria(inspection.qualityGate.criteria, criterionFilter) : []),
    [criterionFilter, inspection],
  );
  const issueCriteriaCount = useMemo(
    () => (inspection ? filterInspectionCriteria(inspection.qualityGate.criteria, 'issues').length : 0),
    [inspection],
  );
  const visibleMarkers = useMemo(
    () => (inspection ? filterInspectionMarkers(inspection.markers, markerFilter) : []),
    [inspection, markerFilter],
  );
  const effectiveLoadedFileName = loadedReportFileName ?? loadedFileName;
  const undoAvailable = activeDoc ? canUndo(activeDoc) : false;
  const redoAvailable = activeDoc ? canRedo(activeDoc) : false;

  const handleReportFilePicked = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    try {
      const text = await file.text();
      const parsed = parseStackReportInspection(text);
      if (!parsed) {
        setLoadError(t('modals.imageStack.inspection.invalidReport'));
        return;
      }
      setLoadError(null);
      setLockError(null);
      setSidecarNotice(null);
      setConfirmStaleExport(false);
      setLoadedFileName(file.name);
      onLoadInspection(parsed, file.name);
      onSourceLockDocChange(buildInitialSidecarFromInspection(parsed));
    } catch (readError) {
      setLoadError(readError instanceof Error ? readError.message : String(readError));
    }
  };

  const handleSidecarFilePicked = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || !baselineDoc) return;
    try {
      const text = await file.text();
      const reconciled = importAndReconcileSidecar(text, {
        runId: baselineDoc.run_id,
        inputManifestSha256: baselineDoc.input_manifest_sha256,
        coordinateFrame: baselineDoc.coordinate_frame,
        sourceBindings: baselineDoc.source_bindings,
      });
      if (!reconciled) {
        setLockError(t('modals.imageStack.inspection.invalidSidecar'));
        return;
      }
      setLockError(null);
      setConfirmStaleExport(false);
      setSidecarNotice(
        reconciled.newlyStaleCount > 0 || reconciled.mismatchReasons.length > 0
          ? t('modals.imageStack.inspection.sidecarImportedStale', {
              stale: reconciled.doc.locks.filter((lock) => lock.status !== 'active').length,
            })
          : t('modals.imageStack.inspection.sidecarImported', {
              locks: reconciled.doc.locks.length,
              reviews: reconciled.doc.manual_review_regions.length,
            }),
      );
      onSourceLockDocChange(reconciled.doc);
    } catch (readError) {
      setLockError(readError instanceof Error ? readError.message : String(readError));
    }
  };

  const handleRevealReport = () => {
    if (!stackReportPath) return;
    void invoke(Invokes.ShowInFinder, { path: stackReportPath }).catch(() => {});
  };

  const handleExportDiagnosticSummary = () => {
    if (!inspection) return;
    const summary = buildDiagnosticSummaryExport(inspection, exportSummary);
    const safeRunId = inspection.runId.replace(/[^a-zA-Z0-9._-]+/g, '_') || 'stack-run';
    downloadTextFile(
      `diagnostic-summary-${safeRunId}.json`,
      `${JSON.stringify(summary, null, 2)}\n`,
      'application/json',
    );
  };

  const toggleAllowedSourceId = (sourceId: number) => {
    setSelectedAllowedSourceIds((previous) =>
      previous.includes(sourceId)
        ? previous.filter((id) => id !== sourceId)
        : [...previous, sourceId].sort((a, b) => a - b),
    );
  };

  const handleAddSourceLock = () => {
    if (!activeDoc || activeDoc.source_bindings.length === 0) return;
    const validIds = new Set(activeDoc.source_bindings.map((binding) => binding.source_id));
    const requestedSources =
      lockPolicy === 'single_source'
        ? [validIds.has(selectedSourceId) ? selectedSourceId : activeDoc.source_bindings[0].source_id]
        : selectedAllowedSourceIds.filter((id) => validIds.has(id));
    if (requestedSources.length === 0) {
      setLockError(t('modals.imageStack.inspection.multiSourceHint'));
      return;
    }
    try {
      const next = commitAddSourceLock(
        activeDoc,
        {
          base_run_id: activeDoc.run_id,
          coordinate_frame_id: activeDoc.coordinate_frame.id,
          expected_head: activeDoc.history.head,
        },
        {
          polygonWorld: targetRegionInfo.polygon,
          requestedSources,
          policy: lockPolicy,
          evidence: {
            coordinate_frame_id: activeDoc.coordinate_frame.id,
            input_manifest_sha256: activeDoc.input_manifest_sha256,
            eligible_tiles: Math.max(1, Math.ceil(targetRegionInfo.areaPx / (256 * 256))),
            covered_tiles: Math.max(1, Math.ceil(targetRegionInfo.areaPx / (256 * 256))),
            has_no_owner_or_transparent: false,
            footprint_and_inverse_valid: true,
            integrity_errors: [],
            coverage_owner_tile_digest: sha256Hex(
              `${activeDoc.run_id}:${requestedSources.join(',')}:${targetRegionInfo.box.x},${targetRegionInfo.box.y},${targetRegionInfo.box.width},${targetRegionInfo.box.height}`,
            ),
          },
        },
      );
      setLockError(null);
      setSidecarNotice(null);
      onSourceLockDocChange(next);
    } catch (err) {
      setLockError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleAddManualReview = () => {
    if (!activeDoc) return;
    try {
      const next = commitAddManualReviewRegion(
        activeDoc,
        {
          base_run_id: activeDoc.run_id,
          coordinate_frame_id: activeDoc.coordinate_frame.id,
          expected_head: activeDoc.history.head,
        },
        {
          polygonWorld: targetRegionInfo.polygon,
          reason: reviewReason || 'alignment_review',
        },
      );
      setLockError(null);
      setSidecarNotice(null);
      onSourceLockDocChange(next);
    } catch (err) {
      setLockError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleRemoveLock = (lockId: string) => {
    if (!activeDoc) return;
    try {
      const next = commitRemoveSourceLock(
        activeDoc,
        {
          base_run_id: activeDoc.run_id,
          coordinate_frame_id: activeDoc.coordinate_frame.id,
          expected_head: activeDoc.history.head,
        },
        lockId,
      );
      setLockError(null);
      onSourceLockDocChange(next);
    } catch (err) {
      setLockError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleRemoveReviewRegion = (regionId: string) => {
    if (!activeDoc) return;
    try {
      const next = commitRemoveManualReviewRegion(
        activeDoc,
        {
          base_run_id: activeDoc.run_id,
          coordinate_frame_id: activeDoc.coordinate_frame.id,
          expected_head: activeDoc.history.head,
        },
        regionId,
      );
      setLockError(null);
      onSourceLockDocChange(next);
    } catch (err) {
      setLockError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUndoLockEvent = () => {
    if (!activeDoc || !undoAvailable) return;
    try {
      const next = commitUndo(activeDoc, {
        base_run_id: activeDoc.run_id,
        coordinate_frame_id: activeDoc.coordinate_frame.id,
        expected_head: activeDoc.history.head,
      });
      setLockError(null);
      onSourceLockDocChange(next);
    } catch (err) {
      setLockError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleRedoLockEvent = () => {
    if (!activeDoc || !redoAvailable) return;
    try {
      const next = commitRedo(activeDoc, {
        base_run_id: activeDoc.run_id,
        coordinate_frame_id: activeDoc.coordinate_frame.id,
        expected_head: activeDoc.history.head,
      });
      setLockError(null);
      onSourceLockDocChange(next);
    } catch (err) {
      setLockError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleExportSidecar = () => {
    if (!activeDoc || !exportSummary) return;
    if (exportSummary.requiresStaleConfirmation && !confirmStaleExport) {
      setLockError(t('modals.imageStack.inspection.staleWarning'));
      return;
    }
    const { sidecarJson, eventsJsonl } = serializeSidecarFiles(activeDoc);
    downloadTextFile('source-locks.json', sidecarJson, 'application/json');
    downloadTextFile('source-locks.events.jsonl', eventsJsonl, 'application/x-ndjson');
  };

  const sidecarDigest = exportSummary ? exportSummary.sidecarSha256.slice(0, 12) : null;
  const effectiveAuditTiffPath = auditTiffPath || exportedTiffPath || inspection?.output.path || '';
  const effectiveAuditSidecarSha = auditRecordedSidecarSha || exportSummary?.sidecarSha256 || '';

  const pickSingleFilePath = async (title: string, extensions: string[]): Promise<string | null> => {
    const selected = await open({
      directory: false,
      multiple: false,
      title,
      filters: [{ name: title, extensions }],
    });
    return typeof selected === 'string' ? selected : null;
  };

  const handlePickAuditTiff = async () => {
    try {
      const selected = await pickSingleFilePath(t('modals.imageStack.inspection.pickTiffFile'), ['tif', 'tiff']);
      if (selected) {
        setAuditTiffPath(selected);
        setBundleAuditError(null);
      }
    } catch (pickError) {
      setBundleAuditError(pickError instanceof Error ? pickError.message : String(pickError));
    }
  };

  const handlePickAuditSidecar = async () => {
    try {
      const selected = await pickSingleFilePath(t('modals.imageStack.inspection.pickSidecarFile'), ['json']);
      if (selected) {
        setAuditSidecarPath(selected);
        setBundleAuditError(null);
      }
    } catch (pickError) {
      setBundleAuditError(pickError instanceof Error ? pickError.message : String(pickError));
    }
  };

  const handlePickAuditEvents = async () => {
    try {
      const selected = await pickSingleFilePath(t('modals.imageStack.inspection.pickEventsFile'), ['jsonl', 'ndjson']);
      if (selected) {
        setAuditEventsPath(selected);
        setBundleAuditError(null);
      }
    } catch (pickError) {
      setBundleAuditError(pickError instanceof Error ? pickError.message : String(pickError));
    }
  };

  const handleVerifyExportBundle = async () => {
    if (!effectiveAuditTiffPath.trim() || !auditSidecarPath.trim()) {
      setBundleAuditError(t('modals.imageStack.inspection.bundlePathRequired'));
      return;
    }
    setIsAuditingBundle(true);
    setBundleAuditError(null);
    try {
      const request = buildVerifyExportBundleRequest({
        tiffPath: effectiveAuditTiffPath,
        sidecarPath: auditSidecarPath,
        eventsJsonlPath: auditEventsPath || null,
        recordedTiffSha256: auditRecordedTiffSha,
        recordedSidecarSha256: effectiveAuditSidecarSha,
        currentRunDoc: auditCheckCurrentRun ? baselineDoc : null,
      });
      const result = await invoke<RustExportBindingAudit>(Invokes.VerifySourceLockExportBundle, { ...request });
      setBundleAudit(result);
    } catch (auditError) {
      setBundleAudit(null);
      setBundleAuditError(auditError instanceof Error ? auditError.message : String(auditError));
    } finally {
      setIsAuditingBundle(false);
    }
  };

  return (
    <div className="space-y-4 p-4">
      <section className="space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">{t('modals.imageStack.inspection.title')}</h3>
            <p className="mt-0.5 text-[11px] text-text-secondary">{t('modals.imageStack.inspection.subtitle')}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          <button
            className="flex items-center gap-1.5 rounded-lg bg-card-active px-2.5 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-accent/15 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            onClick={() => fileInputRef.current?.click()}
            type="button"
          >
            <FileSearch aria-hidden="true" size={14} />
            {t('modals.imageStack.inspection.loadReport')}
          </button>
          {stackReportPath && onReloadCurrentReport && (
            <button
              className="flex items-center gap-1.5 rounded-lg bg-card-active px-2.5 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              onClick={onReloadCurrentReport}
              type="button"
            >
              <RefreshCw aria-hidden="true" size={14} />
              {t('modals.imageStack.inspection.reloadCurrentReport')}
            </button>
          )}
          {stackReportPath && (
            <button
              className="flex items-center gap-1.5 rounded-lg bg-card-active px-2.5 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              onClick={handleRevealReport}
              type="button"
            >
              <FolderOpen aria-hidden="true" size={14} />
              {t('modals.imageStack.inspection.revealReport')}
            </button>
          )}
          {inspection && (
            <button
              className="flex items-center gap-1.5 rounded-lg bg-card-active px-2.5 py-1.5 text-xs font-medium text-text-secondary transition-colors hover:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              onClick={handleExportDiagnosticSummary}
              type="button"
            >
              <Download aria-hidden="true" size={14} />
              {t('modals.imageStack.inspection.exportDiagnosticSummary')}
            </button>
          )}
          <input
            accept=".json,application/json"
            className="hidden"
            onChange={(event) => void handleReportFilePicked(event)}
            ref={fileInputRef}
            type="file"
          />
        </div>
        {effectiveLoadedFileName && (
          <p className="truncate text-[10px] text-text-secondary" title={effectiveLoadedFileName}>
            {effectiveLoadedFileName}
          </p>
        )}
        {loadError && (
          <p className="rounded-md bg-status-error/10 p-2 text-[11px] text-status-error" role="alert">
            {loadError}
          </p>
        )}
      </section>

      {!inspection ? (
        <section className="rounded-lg border border-border-color bg-bg-primary/35 p-3 text-xs leading-relaxed text-text-secondary">
          <p>{t('modals.imageStack.inspection.emptyHint')}</p>
          {error && <p className="mt-2 break-words text-[11px] text-status-error">{error}</p>}
        </section>
      ) : (
        <>
          <section className="space-y-2 rounded-lg border border-border-color bg-bg-primary/35 p-3">
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-xs font-semibold text-text-primary">
                {inspection.verdict === 'success' ? (
                  <CheckCircle2 aria-hidden="true" className="text-status-success" size={15} />
                ) : inspection.verdict === 'degraded' ? (
                  <AlertTriangle aria-hidden="true" className="text-status-warning" size={15} />
                ) : (
                  <XCircle aria-hidden="true" className="text-status-error" size={15} />
                )}
                {`${t('modals.imageStack.inspection.verdictLabel')}: ${inspection.verdict}`}
              </span>
              <span className="rounded bg-card-active px-1.5 py-0.5 text-[10px] text-text-secondary">
                {inspection.selectedPath}
              </span>
            </div>
            <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[11px]">
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.qualityGate')}</dt>
                <dd className="font-medium text-text-primary">
                  {`${inspection.qualityGate.verdict} (${inspection.qualityGate.roiCount} ROI)`}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.closure')}</dt>
                <dd className="font-medium text-text-primary">
                  {`${inspection.closure.status}${
                    inspection.closure.maxDirectPairP95Px !== null
                      ? ` · P95 ${inspection.closure.maxDirectPairP95Px.toFixed(2)}px`
                      : ''
                  }`}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.density')}</dt>
                <dd className="font-medium text-text-primary">
                  {`${inspection.outputDensity.status}${
                    inspection.outputDensity.factor !== null ? ` (${inspection.outputDensity.factor.toFixed(3)}×)` : ''
                  }`}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.tone')}</dt>
                <dd className="font-medium text-text-primary">
                  {`${inspection.tone.status}${
                    inspection.tone.maxDeltaE00 !== null ? ` · ΔE ${inspection.tone.maxDeltaE00.toFixed(2)}` : ''
                  }`}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.stations')}</dt>
                <dd className="font-medium text-text-primary">
                  {`${inspection.stationCount} · ${inspection.sources.length - inspection.excludedCount}/${inspection.sources.length}`}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.resources')}</dt>
                <dd className="font-medium text-text-primary">
                  {`${formatGiB(inspection.resources.peakRssBytes)} · ${formatSeconds(inspection.resources.totalSeconds)}`}
                </dd>
              </div>
            </dl>
          </section>

          {inspection.issues.length > 0 && (
            <section className="space-y-2 border-t border-border-color pt-3">
              <h4 className="flex items-center gap-1.5 text-xs font-semibold text-text-primary">
                <ShieldAlert aria-hidden="true" size={14} />
                {t('modals.imageStack.inspection.issuesTitle', { total: inspection.issues.length })}
              </h4>
              <ul className="space-y-1.5">
                {inspection.issues.slice(0, 8).map((issue) => (
                  <li
                    className="rounded-lg border border-border-color bg-bg-primary/30 p-2 text-[11px]"
                    key={issue.code}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`font-medium ${
                          issue.severity === 'error' ? 'text-status-error' : 'text-status-warning'
                        }`}
                      >
                        {issue.summary}
                      </span>
                      <span className="shrink-0 rounded bg-card-active px-1.5 py-0.5 text-[10px] tabular-nums text-text-secondary">
                        {`×${issue.count}`}
                      </span>
                    </div>
                    <p className="mt-0.5 leading-relaxed text-text-secondary">{issue.detail}</p>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {inspection.qualityGate.criteria.length > 0 && (
            <section className="space-y-2 border-t border-border-color pt-3">
              <div className="flex items-center justify-between gap-2">
                <h4 className="text-xs font-semibold text-text-primary">
                  {t('modals.imageStack.inspection.criteriaTitle')}
                </h4>
                <div className="flex items-center gap-1">
                  <button
                    className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                      criterionFilter === 'all'
                        ? 'bg-accent/15 text-text-primary'
                        : 'bg-card-active text-text-secondary hover:text-text-primary'
                    }`}
                    onClick={() => setCriterionFilter('all')}
                    type="button"
                  >
                    {t('modals.imageStack.inspection.filterAll')}
                  </button>
                  <button
                    className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                      criterionFilter === 'issues'
                        ? 'bg-status-error/15 text-status-error'
                        : 'bg-card-active text-text-secondary hover:text-text-primary'
                    }`}
                    onClick={() => setCriterionFilter('issues')}
                    type="button"
                  >
                    {t('modals.imageStack.inspection.filterIssues', { total: issueCriteriaCount })}
                  </button>
                </div>
              </div>
              {visibleCriteria.length === 0 ? (
                <p className="rounded-md bg-bg-primary/25 px-2 py-1.5 text-[11px] text-text-secondary">
                  {t('modals.imageStack.inspection.noMatchingCriteria')}
                </p>
              ) : (
                <ul className="space-y-1">
                  {visibleCriteria.map((criterion) => (
                    <li
                      className="flex items-center justify-between gap-2 rounded-md bg-bg-primary/30 px-2 py-1.5 text-[11px]"
                      key={criterion.name}
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium text-text-primary" title={criterion.name}>
                          {criterion.name}
                        </p>
                        <p className="text-[10px] tabular-nums text-text-secondary">
                          {`${criterion.worstValue !== null ? criterion.worstValue.toFixed(3) : '—'}${
                            criterion.threshold !== null ? ` / ${criterion.threshold}` : ''
                          } · n=${criterion.measurableCount}`}
                        </p>
                      </div>
                      <span
                        className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${criterionBadgeClass(criterion.verdict)}`}
                      >
                        {criterion.verdict}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {inspection.markers.length > 0 && (
            <section className="space-y-2 border-t border-border-color pt-3">
              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <h4 className="flex items-center gap-1.5 text-xs font-semibold text-text-primary">
                  <Crosshair aria-hidden="true" size={14} />
                  {t('modals.imageStack.inspection.markersTitle', { total: visibleMarkers.length })}
                </h4>
                <div className="flex flex-wrap items-center gap-1">
                  {(
                    [
                      ['all', t('modals.imageStack.inspection.filterAll')],
                      ['qg_fail', t('modals.imageStack.inspection.filterMarkerQgFail')],
                      ['qg_unmeasurable', t('modals.imageStack.inspection.filterMarkerUnmeasurable')],
                      ['tone_violation', t('modals.imageStack.inspection.filterMarkerTone')],
                    ] as Array<[MarkerFilter, string]>
                  ).map(([kind, label]) => (
                    <button
                      className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                        markerFilter === kind
                          ? 'bg-accent/15 text-text-primary'
                          : 'bg-card-active text-text-secondary hover:text-text-primary'
                      }`}
                      key={kind}
                      onClick={() => setMarkerFilter(kind)}
                      type="button"
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              {visibleMarkers.length === 0 ? (
                <p className="rounded-md bg-bg-primary/25 px-2 py-1.5 text-[11px] text-text-secondary">
                  {t('modals.imageStack.inspection.noMatchingMarkers')}
                </p>
              ) : (
                <ul className="max-h-44 space-y-1 overflow-y-auto pr-0.5">
                  {visibleMarkers.map((marker) => {
                    const isSelected = marker.id === selectedMarker?.id;
                    return (
                      <li key={marker.id}>
                        <button
                          className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-[11px] transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${
                            isSelected
                              ? 'bg-accent/15 text-text-primary ring-1 ring-inset ring-accent'
                              : 'bg-bg-primary/30 text-text-secondary hover:bg-card-active hover:text-text-primary'
                          }`}
                          onClick={() => onFocusMarker(marker)}
                          type="button"
                        >
                          <span className="min-w-0 flex-1 truncate" title={marker.detail}>
                            {marker.label}
                          </span>
                          <span
                            className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                              marker.kind === 'qg_fail'
                                ? 'bg-status-error/15 text-status-error'
                                : 'bg-status-warning/15 text-status-warning'
                            }`}
                          >
                            {`${Math.round(marker.worldBox.x)},${Math.round(marker.worldBox.y)}`}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </section>
          )}

          <section className="space-y-2 border-t border-border-color pt-3">
            <button
              className="flex w-full items-center justify-between rounded-md bg-bg-primary/30 px-2 py-1.5 text-xs font-semibold text-text-primary transition-colors hover:bg-card-active"
              onClick={() => setShowSources((visible) => !visible)}
              type="button"
            >
              <span>
                {t('modals.imageStack.inspection.sourcesTitle', {
                  active: inspection.sources.length - inspection.excludedCount,
                  total: inspection.sources.length,
                })}
              </span>
              <span className="text-[10px] font-normal text-text-secondary">{showSources ? '−' : '+'}</span>
            </button>
            {showSources && (
              <ul className="max-h-40 space-y-1 overflow-y-auto pr-0.5 text-[11px]">
                {inspection.sources.map((source) => (
                  <li
                    className="flex items-center justify-between gap-2 rounded bg-bg-primary/25 px-2 py-1"
                    key={`${source.index}-${source.path}`}
                  >
                    <span className="min-w-0 flex-1 truncate text-text-primary" title={source.path}>
                      {source.fileName}
                    </span>
                    <span
                      className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] ${
                        source.excluded
                          ? 'bg-status-warning/15 text-status-warning'
                          : 'bg-status-success/15 text-status-success'
                      }`}
                    >
                      {source.excluded ? (source.exclusionReason ?? 'excluded') : 'ok'}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {activeDoc && exportSummary && (
            <section className="space-y-2.5 border-t border-border-color pt-3">
              <div className="flex items-center justify-between gap-2">
                <h4 className="flex items-center gap-1.5 text-xs font-semibold text-text-primary">
                  <Lock aria-hidden="true" size={14} />
                  {t('modals.imageStack.inspection.sourceLocksTitle')}
                </h4>
                {sidecarDigest && (
                  <span className="font-mono text-[10px] text-text-secondary" title={exportSummary.sidecarSha256}>
                    {`#${sidecarDigest}`}
                  </span>
                )}
              </div>
              <p className="text-[11px] leading-relaxed text-text-secondary">
                {t('modals.imageStack.inspection.sourceLocksHint')}
              </p>

              <div className="space-y-1.5 rounded-lg border border-border-color bg-bg-primary/35 p-2.5 text-[11px]">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium text-text-secondary">
                    {`${t('modals.imageStack.inspection.targetRegionLabel')}: ${
                      targetRegionInfo.kind === 'drawn'
                        ? t('modals.imageStack.inspection.drawnTarget')
                        : targetRegionInfo.kind === 'marker'
                          ? t('modals.imageStack.inspection.markerTarget')
                          : t('modals.imageStack.inspection.canvasDefaultTarget')
                    }`}
                  </span>
                  <div className="flex items-center gap-1">
                    {onToggleDrawRegion && (
                      <button
                        aria-pressed={isDrawingRegion}
                        className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                          isDrawingRegion
                            ? 'bg-accent text-white'
                            : 'bg-card-active text-text-primary hover:bg-accent/15'
                        }`}
                        onClick={onToggleDrawRegion}
                        type="button"
                      >
                        <SquareDashedMousePointer aria-hidden="true" size={11} />
                        {isDrawingRegion
                          ? t('modals.imageStack.inspection.drawingRegion')
                          : t('modals.imageStack.inspection.drawRegion')}
                      </button>
                    )}
                    {draftWorldBox && onClearDraftWorldBox && (
                      <button
                        className="rounded bg-card-active px-1.5 py-0.5 text-[10px] text-text-secondary hover:text-text-primary"
                        onClick={onClearDraftWorldBox}
                        type="button"
                      >
                        {t('modals.imageStack.inspection.clearDrawnRegion')}
                      </button>
                    )}
                  </div>
                </div>
                <p className="font-mono text-[10px] tabular-nums text-text-secondary">
                  {`(${Math.round(targetRegionInfo.box.x)}, ${Math.round(targetRegionInfo.box.y)}) · ${Math.round(targetRegionInfo.box.width)}×${Math.round(targetRegionInfo.box.height)} px · ${targetRegionInfo.areaPx.toLocaleString()} px² · ${activeDoc.coordinate_frame.id}`}
                </p>
              </div>

              {activeDoc.source_bindings.length > 0 && (
                <div className="space-y-2 rounded-lg border border-border-color bg-bg-primary/30 p-2.5">
                  <div className="flex gap-1.5">
                    <button
                      className={`flex-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                        lockPolicy === 'single_source'
                          ? 'bg-accent/15 text-text-primary ring-1 ring-inset ring-accent'
                          : 'bg-card-active text-text-secondary hover:text-text-primary'
                      }`}
                      onClick={() => setLockPolicy('single_source')}
                      type="button"
                    >
                      {t('modals.imageStack.inspection.singleSourcePolicy')}
                    </button>
                    <button
                      className={`flex-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                        lockPolicy === 'allowed_sources'
                          ? 'bg-accent/15 text-text-primary ring-1 ring-inset ring-accent'
                          : 'bg-card-active text-text-secondary hover:text-text-primary'
                      }`}
                      onClick={() => setLockPolicy('allowed_sources')}
                      type="button"
                    >
                      {t('modals.imageStack.inspection.allowedSourcesPolicy')}
                    </button>
                  </div>

                  {lockPolicy === 'single_source' ? (
                    <label className="block text-[10px] font-medium text-text-secondary">
                      {t('modals.imageStack.inspection.lockSourceLabel')}
                      <select
                        className="mt-1 w-full rounded-md bg-surface px-2 py-1 text-xs text-text-primary"
                        onChange={(event) => setSelectedSourceId(Number(event.target.value))}
                        value={selectedSourceId}
                      >
                        {activeDoc.source_bindings.map((binding) => (
                          <option key={binding.source_id} value={binding.source_id}>
                            {`#${binding.source_id} · ${fileBaseName(binding.path)}`}
                          </option>
                        ))}
                      </select>
                    </label>
                  ) : (
                    <div className="space-y-1">
                      <p className="text-[10px] text-text-secondary">
                        {t('modals.imageStack.inspection.multiSourceHint')}
                      </p>
                      <div className="max-h-28 space-y-1 overflow-y-auto rounded-md bg-surface/80 p-1.5">
                        {activeDoc.source_bindings.map((binding) => {
                          const checked = selectedAllowedSourceIds.includes(binding.source_id);
                          return (
                            <label
                              className="flex cursor-pointer items-center gap-1.5 rounded px-1 py-0.5 text-[11px] text-text-primary hover:bg-card-active"
                              key={binding.source_id}
                            >
                              <input
                                checked={checked}
                                onChange={() => toggleAllowedSourceId(binding.source_id)}
                                type="checkbox"
                              />
                              <span className="truncate" title={binding.path}>
                                {`#${binding.source_id} · ${fileBaseName(binding.path)}`}
                              </span>
                            </label>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-1.5">
                    <button
                      className="flex items-center gap-1 rounded-md bg-accent/15 px-2.5 py-1 text-[11px] font-medium text-text-primary transition-colors hover:bg-accent/25"
                      onClick={handleAddSourceLock}
                      type="button"
                    >
                      <Lock aria-hidden="true" size={12} />
                      {t('modals.imageStack.inspection.addLock')}
                    </button>
                    <button
                      className="flex items-center gap-1 rounded-md bg-card-active px-2.5 py-1 text-[11px] font-medium text-text-primary transition-colors hover:bg-accent/15"
                      onClick={handleAddManualReview}
                      type="button"
                    >
                      <Crosshair aria-hidden="true" size={12} />
                      {t('modals.imageStack.inspection.addManualReview')}
                    </button>
                  </div>

                  <input
                    aria-label={t('modals.imageStack.inspection.reviewReasonPlaceholder')}
                    className="w-full rounded-md bg-surface px-2 py-1 text-[11px] text-text-primary"
                    onChange={(event) => setReviewReason(event.target.value)}
                    placeholder={t('modals.imageStack.inspection.reviewReasonPlaceholder')}
                    type="text"
                    value={reviewReason}
                  />
                </div>
              )}

              {activeDoc.locks.length > 0 && (
                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-text-secondary">
                    {t('modals.imageStack.inspection.locksListTitle', { total: activeDoc.locks.length })}
                  </p>
                  <ul className="max-h-36 space-y-1 overflow-y-auto pr-0.5 text-[11px]">
                    {activeDoc.locks.map((lock) => {
                      const sourceLabel = lock.requested_sources
                        .map((id) => `#${id} ${sourceNameById.get(id) ?? ''}`.trim())
                        .join(', ');
                      const [bx, by, bw, bh] = lock.region.bbox;
                      return (
                        <li
                          className="flex items-start justify-between gap-1.5 rounded-md bg-bg-primary/30 px-2 py-1.5"
                          key={lock.lock_id}
                        >
                          <button
                            className="min-w-0 flex-1 text-left"
                            onClick={() => onFocusWorldBox?.({ x: bx, y: by, width: bw, height: bh })}
                            type="button"
                          >
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-[10px] font-semibold text-text-primary">
                                {lock.lock_id}
                              </span>
                              <span
                                className={`rounded px-1 py-0.25 text-[9px] font-medium ${
                                  lock.status === 'active'
                                    ? 'bg-status-success/15 text-status-success'
                                    : 'bg-status-warning/15 text-status-warning'
                                }`}
                              >
                                {lock.status}
                              </span>
                            </div>
                            <p className="truncate text-[10px] text-text-secondary" title={sourceLabel}>
                              {sourceLabel}
                            </p>
                            {lock.stale_reason && (
                              <p className="truncate text-[10px] text-status-warning" title={lock.stale_reason}>
                                {lock.stale_reason}
                              </p>
                            )}
                          </button>
                          <button
                            aria-label={`${t('modals.imageStack.inspection.deleteRegion')} ${lock.lock_id}`}
                            className="shrink-0 rounded p-1 text-text-secondary transition-colors hover:bg-status-error/15 hover:text-status-error"
                            onClick={() => handleRemoveLock(lock.lock_id)}
                            type="button"
                          >
                            <Trash2 aria-hidden="true" size={12} />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}

              {activeDoc.manual_review_regions.length > 0 && (
                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-text-secondary">
                    {t('modals.imageStack.inspection.reviewsListTitle', {
                      total: activeDoc.manual_review_regions.length,
                    })}
                  </p>
                  <ul className="max-h-32 space-y-1 overflow-y-auto pr-0.5 text-[11px]">
                    {activeDoc.manual_review_regions.map((region) => {
                      const [bx, by, bw, bh] = computeWorldRegion(region.polygon_world).bbox;
                      return (
                        <li
                          className="flex items-start justify-between gap-1.5 rounded-md bg-bg-primary/30 px-2 py-1.5"
                          key={region.region_id}
                        >
                          <button
                            className="min-w-0 flex-1 text-left"
                            onClick={() => onFocusWorldBox?.({ x: bx, y: by, width: bw, height: bh })}
                            type="button"
                          >
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-[10px] font-semibold text-text-primary">
                                {region.region_id}
                              </span>
                              <span className="rounded bg-violet-500/15 px-1 py-0.25 text-[9px] text-violet-300">
                                {t('modals.imageStack.inspection.manualExcludedTag')}
                              </span>
                            </div>
                            <p className="truncate text-[10px] text-text-secondary" title={region.reason}>
                              {region.reason}
                            </p>
                          </button>
                          <button
                            aria-label={`${t('modals.imageStack.inspection.deleteRegion')} ${region.region_id}`}
                            className="shrink-0 rounded p-1 text-text-secondary transition-colors hover:bg-status-error/15 hover:text-status-error"
                            onClick={() => handleRemoveReviewRegion(region.region_id)}
                            type="button"
                          >
                            <Trash2 aria-hidden="true" size={12} />
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}

              {sidecarNotice && (
                <p className="rounded bg-accent/10 p-2 text-[11px] text-text-primary" role="status">
                  {sidecarNotice}
                </p>
              )}

              {lockError && (
                <p className="rounded bg-status-error/10 p-2 text-[11px] text-status-error" role="alert">
                  {lockError}
                </p>
              )}

              <div className="space-y-1.5 rounded-lg border border-border-color bg-bg-primary/30 p-2.5 text-[10px] text-text-secondary">
                <p>
                  {t('modals.imageStack.inspection.activeSummary', {
                    locks: exportSummary.activeLockCount,
                    pixels: exportSummary.activeLockAreaPx.toLocaleString(),
                  })}
                </p>
                {exportSummary.staleOrUnresolvedCount > 0 && (
                  <p className="font-medium text-status-warning">
                    {t('modals.imageStack.inspection.staleSummary', {
                      stale: exportSummary.staleOrUnresolvedCount,
                      pixels: exportSummary.staleOrUnresolvedAreaPx.toLocaleString(),
                    })}
                  </p>
                )}
                <p>
                  {t('modals.imageStack.inspection.sidecarCounts', {
                    locks: activeDoc.locks.length,
                    reviews: activeDoc.manual_review_regions.length,
                    events: activeDoc.history.head,
                  })}
                </p>
                {exportSummary.requiresStaleConfirmation && (
                  <label className="mt-1 flex cursor-pointer items-start gap-1.5 rounded bg-status-warning/10 p-1.5 text-[10px] text-status-warning">
                    <input
                      checked={confirmStaleExport}
                      className="mt-0.5"
                      onChange={(event) => setConfirmStaleExport(event.target.checked)}
                      type="checkbox"
                    />
                    <span>
                      {t('modals.imageStack.inspection.confirmStaleExport', {
                        stale: exportSummary.staleOrUnresolvedCount,
                      })}
                    </span>
                  </label>
                )}
              </div>

              <div className="flex flex-wrap items-center justify-between gap-1.5">
                <div className="flex items-center gap-1.5">
                  <button
                    className="flex items-center gap-1 rounded-md bg-card-active px-2 py-1 text-[11px] text-text-primary disabled:opacity-40"
                    disabled={!undoAvailable}
                    onClick={handleUndoLockEvent}
                    type="button"
                  >
                    <Undo2 aria-hidden="true" size={12} />
                    {t('modals.imageStack.inspection.undo')}
                  </button>
                  <button
                    className="flex items-center gap-1 rounded-md bg-card-active px-2 py-1 text-[11px] text-text-primary disabled:opacity-40"
                    disabled={!redoAvailable}
                    onClick={handleRedoLockEvent}
                    type="button"
                  >
                    <Redo2 aria-hidden="true" size={12} />
                    {t('modals.imageStack.inspection.redo')}
                  </button>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    className="flex items-center gap-1 rounded-md bg-card-active px-2 py-1 text-[11px] text-text-primary hover:bg-accent/15"
                    onClick={() => sidecarInputRef.current?.click()}
                    type="button"
                  >
                    <Upload aria-hidden="true" size={12} />
                    {t('modals.imageStack.inspection.importSidecar')}
                  </button>
                  <button
                    className="flex items-center gap-1 rounded-md bg-card-active px-2 py-1 text-[11px] text-text-primary hover:bg-accent/15 disabled:opacity-40"
                    disabled={exportSummary.requiresStaleConfirmation && !confirmStaleExport}
                    onClick={handleExportSidecar}
                    type="button"
                  >
                    <Download aria-hidden="true" size={12} />
                    {t('modals.imageStack.inspection.exportSidecar')}
                  </button>
                </div>
                <input
                  accept=".json,application/json"
                  className="hidden"
                  onChange={(event) => void handleSidecarFilePicked(event)}
                  ref={sidecarInputRef}
                  type="file"
                />
              </div>

              <div className="space-y-2 rounded-lg border border-border-color bg-bg-primary/35 p-2.5 text-[11px]">
                <div className="flex items-center justify-between gap-2">
                  <h5 className="flex items-center gap-1.5 text-[11px] font-semibold text-text-primary">
                    <ShieldCheck aria-hidden="true" size={13} />
                    {t('modals.imageStack.inspection.verifyBundleTitle')}
                  </h5>
                  <button
                    className="rounded-md bg-accent/15 px-2 py-1 text-[10px] font-medium text-text-primary transition-colors hover:bg-accent/25 disabled:opacity-40"
                    disabled={isAuditingBundle || !effectiveAuditTiffPath.trim() || !auditSidecarPath.trim()}
                    onClick={() => void handleVerifyExportBundle()}
                    type="button"
                  >
                    {isAuditingBundle
                      ? t('modals.imageStack.inspection.verifyingBundle')
                      : t('modals.imageStack.inspection.verifyBundleButton')}
                  </button>
                </div>
                <p className="text-[10px] leading-relaxed text-text-secondary">
                  {t('modals.imageStack.inspection.verifyBundleHint')}
                </p>
                <div className="flex flex-wrap gap-1">
                  <button
                    className="rounded bg-card-active px-2 py-1 text-[10px] text-text-primary hover:bg-accent/15"
                    onClick={() => void handlePickAuditTiff()}
                    type="button"
                  >
                    {t('modals.imageStack.inspection.pickTiffFile')}
                  </button>
                  <button
                    className="rounded bg-card-active px-2 py-1 text-[10px] text-text-primary hover:bg-accent/15"
                    onClick={() => void handlePickAuditSidecar()}
                    type="button"
                  >
                    {t('modals.imageStack.inspection.pickSidecarFile')}
                  </button>
                  <button
                    className="rounded bg-card-active px-2 py-1 text-[10px] text-text-secondary hover:bg-accent/15 hover:text-text-primary"
                    onClick={() => void handlePickAuditEvents()}
                    type="button"
                  >
                    {t('modals.imageStack.inspection.pickEventsFile')}
                  </button>
                  {auditEventsPath && (
                    <button
                      className="rounded bg-card-active px-1.5 py-1 text-[10px] text-text-secondary hover:text-text-primary"
                      onClick={() => setAuditEventsPath('')}
                      type="button"
                    >
                      {t('modals.imageStack.inspection.clearOptionalFile')}
                    </button>
                  )}
                </div>
                {(effectiveAuditTiffPath || auditSidecarPath || auditEventsPath) && (
                  <div className="space-y-0.5 font-mono text-[10px] text-text-secondary">
                    {effectiveAuditTiffPath && (
                      <p className="truncate" title={effectiveAuditTiffPath}>
                        {`TIFF: ${fileBaseName(effectiveAuditTiffPath)}`}
                      </p>
                    )}
                    {auditSidecarPath && (
                      <p className="truncate" title={auditSidecarPath}>
                        {`JSON: ${fileBaseName(auditSidecarPath)}`}
                      </p>
                    )}
                    {auditEventsPath && (
                      <p className="truncate" title={auditEventsPath}>
                        {`LOG: ${fileBaseName(auditEventsPath)}`}
                      </p>
                    )}
                  </div>
                )}
                <div className="space-y-1.5">
                  <input
                    aria-label={t('modals.imageStack.inspection.recordedTiffShaPlaceholder')}
                    className="w-full rounded-md bg-surface px-2 py-1 font-mono text-[10px] text-text-primary"
                    onChange={(event) => setAuditRecordedTiffSha(event.target.value)}
                    placeholder={t('modals.imageStack.inspection.recordedTiffShaPlaceholder')}
                    type="text"
                    value={auditRecordedTiffSha}
                  />
                  <input
                    aria-label={t('modals.imageStack.inspection.recordedSidecarShaPlaceholder')}
                    className="w-full rounded-md bg-surface px-2 py-1 font-mono text-[10px] text-text-primary"
                    onChange={(event) => setAuditRecordedSidecarSha(event.target.value)}
                    placeholder={t('modals.imageStack.inspection.recordedSidecarShaPlaceholder')}
                    type="text"
                    value={effectiveAuditSidecarSha}
                  />
                  {baselineDoc && (
                    <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-text-secondary">
                      <input
                        checked={auditCheckCurrentRun}
                        onChange={(event) => setAuditCheckCurrentRun(event.target.checked)}
                        type="checkbox"
                      />
                      <span>{t('modals.imageStack.inspection.checkCurrentRunReconciliation')}</span>
                    </label>
                  )}
                </div>
                {bundleAuditError && (
                  <p className="rounded bg-status-error/10 p-2 text-[10px] text-status-error" role="alert">
                    {bundleAuditError}
                  </p>
                )}
                {bundleAudit && (
                  <div
                    className={`space-y-1 rounded-md p-2 text-[10px] ${
                      bundleAudit.can_publish_as_confirmed
                        ? 'bg-status-success/10 text-status-success'
                        : 'bg-status-error/10 text-status-error'
                    }`}
                    role="status"
                  >
                    <div className="flex items-center justify-between gap-2 font-semibold">
                      <span>
                        {bundleAudit.can_publish_as_confirmed
                          ? t('modals.imageStack.inspection.bundleStatusConfirmed')
                          : t('modals.imageStack.inspection.bundleStatusBlocked')}
                      </span>
                      <span className="rounded bg-bg-primary/40 px-1.5 py-0.5 font-mono text-[9px]">
                        {bundleAudit.status}
                      </span>
                    </div>
                    <p className="text-text-secondary">
                      {t('modals.imageStack.inspection.bundleAuditSummary', {
                        locks: bundleAudit.summary.active_locks,
                        stale: bundleAudit.summary.stale_or_unresolved_locks,
                        reviews: bundleAudit.summary.manual_review_regions,
                        tiles: bundleAudit.summary.locked_eligible_tiles,
                        events: bundleAudit.summary.event_count,
                      })}
                    </p>
                    <div className="flex flex-wrap items-center justify-between gap-1 font-mono text-[9px] text-text-secondary">
                      <span title={bundleAudit.tiff_sha256}>
                        {`${t('modals.imageStack.inspection.observedTiffShaLabel')}: ${bundleAudit.tiff_sha256.slice(0, 16)}…`}
                      </span>
                      {auditRecordedTiffSha.trim().toLowerCase() !== bundleAudit.tiff_sha256 && (
                        <button
                          className="rounded bg-card-active px-1.5 py-0.5 font-sans text-[9px] text-text-primary hover:bg-accent/15"
                          onClick={() => setAuditRecordedTiffSha(bundleAudit.tiff_sha256)}
                          type="button"
                        >
                          {t('modals.imageStack.inspection.useObservedTiffSha')}
                        </button>
                      )}
                    </div>
                    <p className="font-mono text-[9px] text-text-secondary" title={bundleAudit.sidecar_sha256}>
                      {`${t('modals.imageStack.inspection.observedSidecarShaLabel')}: ${bundleAudit.sidecar_sha256.slice(0, 16)}…`}
                    </p>
                    {bundleAudit.reasons.length > 0 && (
                      <ul className="list-disc space-y-0.5 pl-4 text-[10px]">
                        {bundleAudit.reasons.map((reason) => (
                          <li key={reason}>{reason}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
