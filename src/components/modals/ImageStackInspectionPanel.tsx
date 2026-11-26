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
  ShieldAlert,
  Undo2,
  XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../ui/AppProperties';
import type { StackReportInspection, StackReportInspectionMarker } from '../../utils/stackReportInspection';
import { parseStackReportInspection } from '../../utils/stackReportInspection';
import type { SourceLockDocument, SourceLockPolicy } from '../../utils/sourceLockSidecar';
import {
  commitAddManualReviewRegion,
  commitAddSourceLock,
  commitUndo,
  computeSidecarDigest,
  createSourceLockDocument,
  serializeSidecarFiles,
  sha256Hex,
} from '../../utils/sourceLockSidecar';

interface ImageStackInspectionPanelProps {
  error: string | null;
  inspection: StackReportInspection | null;
  selectedMarkerId: string | null;
  sourceLockDoc: SourceLockDocument | null;
  stackReportPath: string | null;
  onFocusMarker(marker: StackReportInspectionMarker): void;
  onLoadInspection(inspection: StackReportInspection, fileName: string): void;
  onSourceLockDocChange(doc: SourceLockDocument | null): void;
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

const downloadTextFile = (fileName: string, content: string, mimeType: string) => {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const buildInitialSidecarFromInspection = (inspection: StackReportInspection): SourceLockDocument => {
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
  error,
  inspection,
  selectedMarkerId,
  sourceLockDoc,
  stackReportPath,
  onFocusMarker,
  onLoadInspection,
  onSourceLockDocChange,
}: ImageStackInspectionPanelProps) {
  const { t } = useTranslation();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadedFileName, setLoadedFileName] = useState<string | null>(null);
  const [showSources, setShowSources] = useState(false);
  const [lockError, setLockError] = useState<string | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState<number>(1);
  const [lockPolicy, setLockPolicy] = useState<SourceLockPolicy>('single_source');
  const [reviewReason, setReviewReason] = useState('alignment_review');

  const activeDoc = useMemo(() => {
    if (!inspection) return null;
    if (sourceLockDoc && sourceLockDoc.run_id === inspection.runId) return sourceLockDoc;
    return buildInitialSidecarFromInspection(inspection);
  }, [inspection, sourceLockDoc]);

  const selectedMarker = useMemo(
    () => inspection?.markers.find((marker) => marker.id === selectedMarkerId) ?? inspection?.markers[0] ?? null,
    [inspection, selectedMarkerId],
  );

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
      setLoadedFileName(file.name);
      onLoadInspection(parsed, file.name);
      onSourceLockDocChange(buildInitialSidecarFromInspection(parsed));
    } catch (readError) {
      setLoadError(readError instanceof Error ? readError.message : String(readError));
    }
  };

  const handleRevealReport = () => {
    if (!stackReportPath) return;
    void invoke(Invokes.ShowInFinder, { path: stackReportPath }).catch(() => {});
  };

  const targetWorldPolygon = (): Array<[number, number]> => {
    const box = selectedMarker?.worldBox ?? {
      x: inspection?.origin.x ?? 0,
      y: inspection?.origin.y ?? 0,
      width: Math.min(512, inspection?.canvas?.width ?? 512),
      height: Math.min(512, inspection?.canvas?.height ?? 512),
    };
    return [
      [Math.round(box.x), Math.round(box.y)],
      [Math.round(box.x + box.width), Math.round(box.y)],
      [Math.round(box.x + box.width), Math.round(box.y + box.height)],
      [Math.round(box.x), Math.round(box.y + box.height)],
    ];
  };

  const handleAddSourceLock = () => {
    if (!activeDoc || activeDoc.source_bindings.length === 0) return;
    const binding =
      activeDoc.source_bindings.find((candidate) => candidate.source_id === selectedSourceId) ??
      activeDoc.source_bindings[0];
    try {
      const next = commitAddSourceLock(
        activeDoc,
        {
          base_run_id: activeDoc.run_id,
          coordinate_frame_id: activeDoc.coordinate_frame.id,
          expected_head: activeDoc.history.head,
        },
        {
          polygonWorld: targetWorldPolygon(),
          requestedSources: [binding.source_id],
          policy: lockPolicy,
          evidence: {
            coordinate_frame_id: activeDoc.coordinate_frame.id,
            input_manifest_sha256: activeDoc.input_manifest_sha256,
            eligible_tiles: 1,
            covered_tiles: 1,
            has_no_owner_or_transparent: false,
            footprint_and_inverse_valid: true,
            integrity_errors: [],
            coverage_owner_tile_digest: sha256Hex(
              `${activeDoc.run_id}:${binding.source_id}:${selectedMarker?.id ?? 'canvas'}`,
            ),
          },
        },
      );
      setLockError(null);
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
          polygonWorld: targetWorldPolygon(),
          reason: reviewReason || 'alignment_review',
        },
      );
      setLockError(null);
      onSourceLockDocChange(next);
    } catch (err) {
      setLockError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleUndoLockEvent = () => {
    if (!activeDoc || activeDoc.history.head === 0) return;
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

  const handleExportSidecar = () => {
    if (!activeDoc) return;
    const { sidecarJson, eventsJsonl } = serializeSidecarFiles(activeDoc);
    downloadTextFile('source-locks.json', sidecarJson, 'application/json');
    downloadTextFile('source-locks.events.jsonl', eventsJsonl, 'application/x-ndjson');
  };

  const sidecarDigest = useMemo(() => (activeDoc ? computeSidecarDigest(activeDoc).slice(0, 12) : null), [activeDoc]);

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
          <input
            accept=".json,application/json"
            className="hidden"
            onChange={(event) => void handleReportFilePicked(event)}
            ref={fileInputRef}
            type="file"
          />
        </div>
        {loadedFileName && (
          <p className="truncate text-[10px] text-text-secondary" title={loadedFileName}>
            {loadedFileName}
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
                {t('modals.imageStack.inspection.verdictLabel')}: {inspection.verdict}
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
                  {inspection.closure.status}
                  {inspection.closure.maxDirectPairP95Px !== null
                    ? ` · P95 ${inspection.closure.maxDirectPairP95Px.toFixed(2)}px`
                    : ''}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.density')}</dt>
                <dd className="font-medium text-text-primary">
                  {inspection.outputDensity.status}
                  {inspection.outputDensity.factor !== null ? ` (${inspection.outputDensity.factor.toFixed(3)}×)` : ''}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.tone')}</dt>
                <dd className="font-medium text-text-primary">
                  {inspection.tone.status}
                  {inspection.tone.maxDeltaE00 !== null ? ` · ΔE ${inspection.tone.maxDeltaE00.toFixed(2)}` : ''}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.stations')}</dt>
                <dd className="font-medium text-text-primary">
                  {inspection.stationCount} · {inspection.sources.length - inspection.excludedCount}/
                  {inspection.sources.length}
                </dd>
              </div>
              <div>
                <dt className="text-text-secondary">{t('modals.imageStack.inspection.resources')}</dt>
                <dd className="font-medium text-text-primary">
                  {formatGiB(inspection.resources.peakRssBytes)} · {formatSeconds(inspection.resources.totalSeconds)}
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
                        ×{issue.count}
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
              <h4 className="text-xs font-semibold text-text-primary">
                {t('modals.imageStack.inspection.criteriaTitle')}
              </h4>
              <ul className="space-y-1">
                {inspection.qualityGate.criteria.map((criterion) => (
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
            </section>
          )}

          {inspection.markers.length > 0 && (
            <section className="space-y-2 border-t border-border-color pt-3">
              <h4 className="flex items-center gap-1.5 text-xs font-semibold text-text-primary">
                <Crosshair aria-hidden="true" size={14} />
                {t('modals.imageStack.inspection.markersTitle', { total: inspection.markers.length })}
              </h4>
              <ul className="max-h-44 space-y-1 overflow-y-auto pr-0.5">
                {inspection.markers.map((marker) => {
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
                          {Math.round(marker.worldBox.x)},{Math.round(marker.worldBox.y)}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
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

          {activeDoc && (
            <section className="space-y-2 border-t border-border-color pt-3">
              <div className="flex items-center justify-between gap-2">
                <h4 className="flex items-center gap-1.5 text-xs font-semibold text-text-primary">
                  <Lock aria-hidden="true" size={14} />
                  {t('modals.imageStack.inspection.sourceLocksTitle')}
                </h4>
                {sidecarDigest && (
                  <span className="font-mono text-[10px] text-text-secondary" title={sidecarDigest}>
                    #{sidecarDigest}
                  </span>
                )}
              </div>
              <p className="text-[11px] leading-relaxed text-text-secondary">
                {t('modals.imageStack.inspection.sourceLocksHint')}
              </p>

              {activeDoc.source_bindings.length > 0 && (
                <div className="space-y-1.5 rounded-lg border border-border-color bg-bg-primary/30 p-2.5">
                  <label className="block text-[10px] font-medium text-text-secondary">
                    {t('modals.imageStack.inspection.lockSourceLabel')}
                    <select
                      className="mt-1 w-full rounded-md bg-surface px-2 py-1 text-xs text-text-primary"
                      onChange={(event) => setSelectedSourceId(Number(event.target.value))}
                      value={selectedSourceId}
                    >
                      {activeDoc.source_bindings.map((binding) => (
                        <option key={binding.source_id} value={binding.source_id}>
                          #{binding.source_id} · {binding.path.split(/[\\/]/).pop() ?? binding.path}
                        </option>
                      ))}
                    </select>
                  </label>

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

              {lockError && (
                <p className="rounded bg-status-error/10 p-2 text-[11px] text-status-error" role="alert">
                  {lockError}
                </p>
              )}

              <div className="flex items-center justify-between gap-2 text-[11px] text-text-secondary">
                <span>
                  {t('modals.imageStack.inspection.sidecarCounts', {
                    locks: activeDoc.locks.length,
                    reviews: activeDoc.manual_review_regions.length,
                    events: activeDoc.history.head,
                  })}
                </span>
                <div className="flex items-center gap-1.5">
                  <button
                    className="flex items-center gap-1 rounded-md bg-card-active px-2 py-1 text-[11px] text-text-primary disabled:opacity-40"
                    disabled={activeDoc.history.head === 0}
                    onClick={handleUndoLockEvent}
                    type="button"
                  >
                    <Undo2 aria-hidden="true" size={12} />
                    {t('modals.imageStack.inspection.undo')}
                  </button>
                  <button
                    className="flex items-center gap-1 rounded-md bg-card-active px-2 py-1 text-[11px] text-text-primary"
                    onClick={handleExportSidecar}
                    type="button"
                  >
                    <Download aria-hidden="true" size={12} />
                    {t('modals.imageStack.inspection.exportSidecar')}
                  </button>
                </div>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
