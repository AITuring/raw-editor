export type StackRunVerdict = 'success' | 'degraded' | 'rejected' | 'cancelled' | 'unknown';

export type QualityGateCriterionVerdict = 'pass' | 'fail' | 'insufficient_evidence' | 'not_applicable' | 'unknown';

export type InspectionMarkerKind = 'qg_fail' | 'qg_unmeasurable' | 'tone_violation';

export interface StackReportCriterionSummary {
  name: string;
  verdict: QualityGateCriterionVerdict;
  threshold: number | null;
  thresholdMin: number | null;
  thresholdMax: number | null;
  measurableCount: number;
  unmeasurableCount: number;
  worstValue: number | null;
  worstWorld: { x: number; y: number } | null;
  worstOwnerPath: string | null;
}

export interface StackReportInspectionMarker {
  id: string;
  kind: InspectionMarkerKind;
  criterion?: string;
  label: string;
  detail: string;
  /** World-space bounding box in native output pixels. */
  worldBox: { x: number; y: number; width: number; height: number };
  /** Normalized 0..1 box on the rendered canvas for preview overlay & zoomToRegion. */
  normalizedBox: { x: number; y: number; width: number; height: number };
  ownerPath?: string | null;
  stations?: [number, number] | null;
  value?: number | null;
}

export interface StackReportSourceEntry {
  index: number;
  path: string;
  fileName: string;
  sha256Before: string;
  sha256After: string;
  decoded: boolean;
  digestIntact: boolean | null;
  excluded: boolean;
  exclusionReason: string | null;
}

export interface StackReportIssueExplanation {
  code: string;
  severity: 'error' | 'warning';
  count: number;
  summary: string;
  detail: string;
}

export interface StackReportInspection {
  schema: number | null;
  runId: string;
  pipelineVersion: string;
  selectedPath: string;
  verdict: StackRunVerdict;
  canvas: { width: number; height: number } | null;
  origin: { x: number; y: number };
  stationCount: number;
  qualityGate: {
    verdict: QualityGateCriterionVerdict;
    roiCount: number;
    unresolvedPixels: number | null;
    criteria: StackReportCriterionSummary[];
  };
  closure: {
    status: string;
    maxDirectPairP95Px: number | null;
    residualMedianPx: number | null;
    residualP95Px: number | null;
    fallbackReason: string | null;
  };
  outputDensity: {
    status: string;
    factor: number | null;
    invalidProbeCount: number;
    bypassReason: string | null;
  };
  tone: {
    status: string;
    maxDeltaE00: number | null;
    threshold: number | null;
    violationCount: number;
  };
  resources: {
    peakRssBytes: number | null;
    memoryThresholdBytes: number | null;
    memoryThresholdExceeded: boolean;
    workerThreads: number | null;
    totalSeconds: number | null;
  };
  output: {
    written: boolean;
    format: string;
    bitDepth: number | null;
    alphaPreserved: boolean | null;
    artifactValidation: string;
    path: string;
    resultId: string;
  };
  sources: StackReportSourceEntry[];
  excludedCount: number;
  issues: StackReportIssueExplanation[];
  markers: StackReportInspectionMarker[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asFiniteNumber = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;

const asString = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback);

const fileBaseName = (filePath: string): string => {
  const clean = filePath.split('?')[0];
  return clean.split(/[\\/]/).pop() || clean;
};

const normalizeVerdict = (value: unknown): StackRunVerdict => {
  if (value === 'success' || value === 'degraded' || value === 'rejected' || value === 'cancelled') {
    return value;
  }
  return 'unknown';
};

const normalizeCriterionVerdict = (value: unknown): QualityGateCriterionVerdict => {
  if (value === 'pass' || value === 'fail' || value === 'insufficient_evidence' || value === 'not_applicable') {
    return value;
  }
  return 'unknown';
};

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));

const normalizeBoxToCanvas = (
  box: { x: number; y: number; width: number; height: number },
  canvas: { width: number; height: number } | null,
  origin: { x: number; y: number },
): { x: number; y: number; width: number; height: number } | null => {
  if (!canvas || canvas.width <= 0 || canvas.height <= 0) return null;
  const relX = box.x - origin.x;
  const relY = box.y - origin.y;
  const x = clamp01(relX / canvas.width);
  const y = clamp01(relY / canvas.height);
  const maxWidth = Math.max(0.004, 1 - x);
  const maxHeight = Math.max(0.004, 1 - y);
  const width = Math.min(maxWidth, Math.max(0.008, box.width / canvas.width));
  const height = Math.min(maxHeight, Math.max(0.008, box.height / canvas.height));
  return { x, y, width, height };
};

const isCriterionMeasurementFailing = (
  criterion: { name: string; threshold: number | null; thresholdMin: number | null; thresholdMax: number | null },
  value: number,
): boolean => {
  if (criterion.thresholdMin !== null && value < criterion.thresholdMin) return true;
  if (criterion.thresholdMax !== null && value > criterion.thresholdMax) return true;
  if (criterion.threshold === null) return false;
  if (criterion.name === 'boundary_stroke_alignment' || criterion.name === 'roi_delta_e00') {
    return value > criterion.threshold;
  }
  if (criterion.thresholdMin === null && criterion.thresholdMax === null) {
    return value < criterion.threshold;
  }
  return false;
};

const pickWorstMeasurement = (
  criterion: { name: string; threshold: number | null; thresholdMin: number | null; thresholdMax: number | null },
  samples: Array<{ value: number; world: { x: number; y: number } | null; ownerPath: string | null }>,
) => {
  if (samples.length === 0) return null;
  const higherIsWorse = criterion.name === 'boundary_stroke_alignment' || criterion.name === 'roi_delta_e00';
  const twoSided = criterion.thresholdMin !== null && criterion.thresholdMax !== null;
  return samples.reduce((worst, candidate) => {
    if (twoSided && criterion.thresholdMin !== null && criterion.thresholdMax !== null) {
      const center = (criterion.thresholdMin + criterion.thresholdMax) * 0.5;
      return Math.abs(candidate.value - center) > Math.abs(worst.value - center) ? candidate : worst;
    }
    if (higherIsWorse) {
      return candidate.value > worst.value ? candidate : worst;
    }
    return candidate.value < worst.value ? candidate : worst;
  });
};

const formatReasonExplanation = (reason: string, count: number): { summary: string; detail: string } => {
  const known: Record<string, { summary: string; detail: string }> = {
    quality_gate_criterion_failed: {
      summary: 'Quality Gate criterion failed',
      detail: 'One or more native quality criteria fell below contract limits; inspect marked ROIs before exporting.',
    },
    closure_unreliable_residual: {
      summary: 'Station closure residual exceeded contract',
      detail: 'Direct-pair reprojection error exceeded the 3.0 px world-pixel limit across camera stations.',
    },
    geometry_disconnected: {
      summary: 'Camera positions could not form a single connected geometry',
      detail: 'Overlapping photos split into disconnected components; check missing bridge shots or framing gaps.',
    },
    tone_boundary_delta_e_exceeded: {
      summary: 'Tone discontinuity across station boundary',
      detail: 'Low-frequency Delta E00 across adjacent owner regions exceeded 1.5; inspect highlighted seam bands.',
    },
    memory_threshold_exceeded: {
      summary: 'Memory gate exceeded',
      detail: 'Process RSS exceeded the configured memory threshold and the run stopped fail-closed.',
    },
    intra_station_registration_failed: {
      summary: 'Intra-station frame registration failed',
      detail: 'One or more focus frames could not be registered to their station anchor and were skipped.',
    },
    intra_station_rejected_from_group: {
      summary: 'Frame excluded from station group',
      detail: 'A frame diverged geometrically from its capture group and was left out of fusion.',
    },
    fusion_single_frame_degraded: {
      summary: 'Single-frame capture station fallback',
      detail: 'One or more stations had only a single usable focus frame instead of a multi-focus stack.',
    },
    fusion_low_sharpness_region: {
      summary: 'Low-sharpness region detected',
      detail: 'Every available photo in the flagged area was softer than the station sharpness baseline.',
    },
    station_relation_scale_out_of_range: {
      summary: 'Station scale ratio out of supported range',
      detail: 'Pairwise magnification between stations fell outside the accepted calibration envelope.',
    },
    topology_index_ambiguous: {
      summary: 'Ambiguous capture topology',
      detail: 'Multiple candidate station relationships competed; geometry relied on verified direct constraints.',
    },
    grouping_accumulated_motion: {
      summary: 'Noticeable camera motion inside a station',
      detail: 'Focus frames shifted relative to each other within a station and required local alignment.',
    },
    tone_gain_clamped: {
      summary: 'Tone harmonization gain clamped',
      detail: 'Exposure difference between neighboring stations hit the safety clamp limit.',
    },
    run_cancelled_by_user: {
      summary: 'Run cancelled',
      detail: 'Processing stopped before publishing final output.',
    },
  };
  const matched = known[reason];
  if (matched) return matched;
  return {
    summary: reason.replaceAll('_', ' '),
    detail: `Recorded ${count} diagnostic event(s) for ${reason}.`,
  };
};

/**
 * Parse a raw `Stack_Report` JSON object (or JSON string) into a normalized
 * inspection view model for alignment review, failure explanation, and
 * preview marker navigation.
 */
export function parseStackReportInspection(raw: unknown): StackReportInspection | null {
  let payload: unknown = raw;
  if (typeof raw === 'string') {
    try {
      payload = JSON.parse(raw);
    } catch {
      return null;
    }
  }
  if (!isRecord(payload)) return null;

  const degradation = isRecord(payload.degradation) ? payload.degradation : {};
  const composition = isRecord(payload.composition) ? payload.composition : {};
  const canvasRaw = isRecord(composition.canvas) ? composition.canvas : {};
  const outputRaw = isRecord(payload.output) ? payload.output : {};
  const publicationCrop = isRecord(composition.publication_crop) ? composition.publication_crop : null;

  const canvasWidth =
    asFiniteNumber(publicationCrop?.width) ?? asFiniteNumber(canvasRaw.width) ?? asFiniteNumber(outputRaw.width);
  const canvasHeight =
    asFiniteNumber(publicationCrop?.height) ?? asFiniteNumber(canvasRaw.height) ?? asFiniteNumber(outputRaw.height);
  const canvas =
    canvasWidth !== null && canvasWidth > 0 && canvasHeight !== null && canvasHeight > 0
      ? { width: Math.round(canvasWidth), height: Math.round(canvasHeight) }
      : null;

  const origin = {
    x: asFiniteNumber(publicationCrop?.x) ?? 0,
    y: asFiniteNumber(publicationCrop?.y) ?? 0,
  };

  const grouping = isRecord(payload.grouping) ? payload.grouping : {};
  const stationCount = Math.max(0, Math.round(asFiniteNumber(grouping.station_count) ?? 0));

  const closureRaw = isRecord(payload.closure) ? payload.closure : {};
  const closure = {
    status: asString(closureRaw.status, 'not_run'),
    maxDirectPairP95Px: asFiniteNumber(closureRaw.max_direct_pair_p95_px),
    residualMedianPx: asFiniteNumber(closureRaw.residual_median_px),
    residualP95Px: asFiniteNumber(closureRaw.residual_p95_px),
    fallbackReason: typeof closureRaw.fallback_reason === 'string' ? closureRaw.fallback_reason : null,
  };

  const densityRaw = isRecord(composition.output_density)
    ? composition.output_density
    : isRecord(payload.output_density)
      ? payload.output_density
      : {};
  const outputDensity = {
    status: asString(densityRaw.status, 'not_run'),
    factor: asFiniteNumber(densityRaw.factor),
    invalidProbeCount: Math.max(0, Math.round(asFiniteNumber(densityRaw.invalid_probe_count) ?? 0)),
    bypassReason:
      typeof densityRaw.bypass_reason === 'string'
        ? densityRaw.bypass_reason
        : typeof densityRaw.reason === 'string'
          ? densityRaw.reason
          : null,
  };

  const toneRaw = isRecord(payload.tone) ? payload.tone : {};
  const boundaryDeltaE = isRecord(toneRaw.boundary_delta_e) ? toneRaw.boundary_delta_e : {};
  const toneViolationsRaw = Array.isArray(boundaryDeltaE.violations) ? boundaryDeltaE.violations : [];
  const tone = {
    status: asString(toneRaw.status, 'not_run'),
    maxDeltaE00: asFiniteNumber(boundaryDeltaE.max),
    threshold: asFiniteNumber(boundaryDeltaE.threshold),
    violationCount: toneViolationsRaw.length,
  };

  const resourcesRaw = isRecord(payload.resources) ? payload.resources : {};
  const performanceRaw = isRecord(payload.performance) ? payload.performance : {};
  const stageSecondsRaw = isRecord(performanceRaw.stage_seconds) ? performanceRaw.stage_seconds : {};
  const resources = {
    peakRssBytes: asFiniteNumber(resourcesRaw.peak_rss_bytes),
    memoryThresholdBytes: asFiniteNumber(resourcesRaw.memory_threshold_bytes),
    memoryThresholdExceeded: resourcesRaw.memory_threshold_exceeded === true,
    workerThreads: asFiniteNumber(resourcesRaw.worker_threads),
    totalSeconds: asFiniteNumber(stageSecondsRaw.total_seconds),
  };

  const output = {
    written: outputRaw.written === true,
    format: asString(outputRaw.format, ''),
    bitDepth: asFiniteNumber(outputRaw.bit_depth),
    alphaPreserved: typeof outputRaw.alpha_preserved === 'boolean' ? outputRaw.alpha_preserved : null,
    artifactValidation: asString(outputRaw.artifact_validation, 'not_run'),
    path: asString(outputRaw.path, ''),
    resultId: asString(outputRaw.result_id, ''),
  };

  const inputRaw = isRecord(payload.input) ? payload.input : {};
  const excludedRaw = Array.isArray(inputRaw.excluded_sources) ? inputRaw.excluded_sources : [];
  const excludedByPath = new Map<string, string>();
  for (const item of excludedRaw) {
    if (!isRecord(item) || typeof item.path !== 'string') continue;
    excludedByPath.set(item.path, asString(item.reason, 'excluded'));
  }

  const sourcesRaw = Array.isArray(inputRaw.sources) ? inputRaw.sources : [];
  const seenPaths = new Set<string>();
  const sources: StackReportSourceEntry[] = [];
  for (const item of sourcesRaw) {
    if (!isRecord(item) || typeof item.path !== 'string') continue;
    seenPaths.add(item.path);
    const shaBefore = asString(item.sha256_before);
    const shaAfter = asString(item.sha256_after);
    const exclusionReason = excludedByPath.get(item.path) ?? null;
    sources.push({
      index: sources.length,
      path: item.path,
      fileName: fileBaseName(item.path),
      sha256Before: shaBefore,
      sha256After: shaAfter,
      decoded: item.decoded !== false,
      digestIntact: shaBefore.length > 0 && shaAfter.length > 0 ? shaBefore === shaAfter : null,
      excluded: exclusionReason !== null || item.decoded === false,
      exclusionReason: exclusionReason ?? (item.decoded === false ? 'decode_failed' : null),
    });
  }
  for (const [excludedPath, reason] of excludedByPath.entries()) {
    if (seenPaths.has(excludedPath)) continue;
    sources.push({
      index: sources.length,
      path: excludedPath,
      fileName: fileBaseName(excludedPath),
      sha256Before: '',
      sha256After: '',
      decoded: false,
      digestIntact: null,
      excluded: true,
      exclusionReason: reason,
    });
  }

  const qgRaw = isRecord(payload.quality_gate) ? payload.quality_gate : {};
  const legacyNativeEvidence = isRecord(qgRaw.legacy_native_evidence) ? qgRaw.legacy_native_evidence : null;
  const reverseLookupFailures = asFiniteNumber(qgRaw.owner_reverse_lookup_failures);
  const unresolvedPixels = asFiniteNumber(legacyNativeEvidence?.unresolved_pixel_count) ?? reverseLookupFailures;
  const criteriaRaw = Array.isArray(qgRaw.criteria) ? qgRaw.criteria : [];
  const criteria: StackReportCriterionSummary[] = [];
  const markers: StackReportInspectionMarker[] = [];

  for (const item of criteriaRaw) {
    if (!isRecord(item) || typeof item.name !== 'string') continue;
    const criterionMeta = {
      name: item.name,
      threshold: asFiniteNumber(item.threshold),
      thresholdMin: asFiniteNumber(item.threshold_min),
      thresholdMax: asFiniteNumber(item.threshold_max),
    };
    const verdict = normalizeCriterionVerdict(item.verdict ?? item.status);
    const measuredRaw = Array.isArray(item.measured) ? item.measured : [];
    const parsedSamples: Array<{ value: number; world: { x: number; y: number } | null; ownerPath: string | null }> =
      [];
    for (const sample of measuredRaw) {
      if (!isRecord(sample)) continue;
      const value = asFiniteNumber(sample.measured ?? sample.value);
      if (value === null) continue;
      const worldRaw = isRecord(sample.world) ? sample.world : null;
      const wx = asFiniteNumber(worldRaw?.x ?? worldRaw?.left);
      const wy = asFiniteNumber(worldRaw?.y ?? worldRaw?.top);
      const world = wx !== null && wy !== null ? { x: wx, y: wy } : null;
      const ownerPath =
        typeof sample.owner_path === 'string' && sample.owner_path.length > 0 ? sample.owner_path : null;
      parsedSamples.push({ value, world, ownerPath });

      if (verdict === 'fail' && world && isCriterionMeasurementFailing(criterionMeta, value)) {
        const roiWidth = asFiniteNumber(worldRaw?.width) ?? 256;
        const roiHeight = asFiniteNumber(worldRaw?.height) ?? 256;
        const worldBox = { x: world.x, y: world.y, width: roiWidth, height: roiHeight };
        const normalizedBox = normalizeBoxToCanvas(worldBox, canvas, origin);
        if (normalizedBox) {
          const ownerLabel = ownerPath ? ` · ${fileBaseName(ownerPath)}` : '';
          markers.push({
            id: `qg-${item.name}-${markers.length}`,
            kind: 'qg_fail',
            criterion: item.name,
            label: `${item.name}: ${value.toFixed(3)}`,
            detail: `${item.name} = ${value.toFixed(4)} at (${Math.round(world.x)}, ${Math.round(world.y)})${ownerLabel}`,
            worldBox,
            normalizedBox,
            ownerPath,
            value,
          });
        }
      }
    }
    const worst = pickWorstMeasurement(criterionMeta, parsedSamples);
    criteria.push({
      name: item.name,
      verdict,
      threshold: criterionMeta.threshold,
      thresholdMin: criterionMeta.thresholdMin,
      thresholdMax: criterionMeta.thresholdMax,
      measurableCount: Math.max(0, Math.round(asFiniteNumber(item.measurable_count) ?? parsedSamples.length)),
      unmeasurableCount: Math.max(0, Math.round(asFiniteNumber(item.unmeasurable_count) ?? 0)),
      worstValue: worst?.value ?? null,
      worstWorld: worst?.world ?? null,
      worstOwnerPath: worst?.ownerPath ?? null,
    });
  }

  for (const violation of toneViolationsRaw) {
    if (!isRecord(violation)) continue;
    const deltaE = asFiniteNumber(violation.delta_e00);
    const worldRaw = isRecord(violation.world) ? violation.world : null;
    const left = asFiniteNumber(worldRaw?.left ?? worldRaw?.x);
    const top = asFiniteNumber(worldRaw?.top ?? worldRaw?.y);
    const width = asFiniteNumber(worldRaw?.width) ?? 256;
    const height = asFiniteNumber(worldRaw?.height) ?? 256;
    if (left === null || top === null) continue;
    const worldBox = { x: left, y: top, width, height };
    const normalizedBox = normalizeBoxToCanvas(worldBox, canvas, origin);
    if (!normalizedBox) continue;
    const stationLeft = asFiniteNumber(violation.left);
    const stationRight = asFiniteNumber(violation.right);
    const stations: [number, number] | null =
      stationLeft !== null && stationRight !== null ? [Math.round(stationLeft), Math.round(stationRight)] : null;
    const stationLabel = stations ? `S${stations[0]}↔S${stations[1]}` : 'boundary';
    markers.push({
      id: `tone-${markers.length}`,
      kind: 'tone_violation',
      label: `ΔE00 ${deltaE !== null ? deltaE.toFixed(2) : '?'} (${stationLabel})`,
      detail: `Tone boundary ΔE00 = ${deltaE !== null ? deltaE.toFixed(3) : '?'} at (${Math.round(left)}, ${Math.round(top)}) ${stationLabel}`,
      worldBox,
      normalizedBox,
      stations,
      value: deltaE,
    });
  }

  const issueCounts = new Map<string, { severity: 'error' | 'warning'; count: number }>();
  const degradationEntries = Array.isArray(degradation.entries) ? degradation.entries : [];
  for (const entry of degradationEntries) {
    if (!isRecord(entry) || typeof entry.reason !== 'string') continue;
    const severity: 'error' | 'warning' =
      entry.severity === 'rejected' || entry.severity === 'error' || entry.severity === 'fatal' ? 'error' : 'warning';
    const existing = issueCounts.get(entry.reason);
    if (existing) {
      existing.count += 1;
      if (severity === 'error') existing.severity = 'error';
    } else {
      issueCounts.set(entry.reason, { severity, count: 1 });
    }
  }
  const issues: StackReportIssueExplanation[] = Array.from(issueCounts.entries())
    .map(([code, meta]) => {
      const explained = formatReasonExplanation(code, meta.count);
      return {
        code,
        severity: meta.severity,
        count: meta.count,
        summary: explained.summary,
        detail: explained.detail,
      };
    })
    .sort((left, right) => {
      if (left.severity !== right.severity) return left.severity === 'error' ? -1 : 1;
      return right.count - left.count || left.code.localeCompare(right.code);
    });

  return {
    schema: asFiniteNumber(payload.schema),
    runId: asString(payload.run_id, 'unknown-run'),
    pipelineVersion: asString(payload.pipeline_version, ''),
    selectedPath: asString(payload.selected_path, 'unknown'),
    verdict: normalizeVerdict(degradation.result),
    canvas,
    origin,
    stationCount,
    qualityGate: {
      verdict: normalizeCriterionVerdict(qgRaw.verdict),
      roiCount: Math.max(0, Math.round(asFiniteNumber(qgRaw.roi_count) ?? 0)),
      unresolvedPixels,
      criteria,
    },
    closure,
    outputDensity,
    tone,
    resources,
    output,
    sources,
    excludedCount: sources.filter((source) => source.excluded).length,
    issues,
    markers,
  };
}
