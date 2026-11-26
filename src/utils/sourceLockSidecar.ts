/**
 * Source-locked retouching sidecar contract (Task 9 / Task 20 Phase 1).
 *
 * Implements the pure data contract, deterministic SHA-256 digest binding,
 * eligibility checks, staleness invalidation, monotonic undo/redo event log
 * with concurrent-session head checks, export binding verification, and
 * manual-review Quality Gate accounting specified in
 * `docs/artwork-lanes/9-source-lock-design.md`.
 *
 * This module never mutates RGB pixels or weakens automatic Quality Gate
 * thresholds.
 */

export const SOURCE_LOCK_SCHEMA_VERSION = 1 as const;

export type SourceLockPolicy = 'single_source' | 'allowed_sources';
export type SourceLockStatus = 'active' | 'stale' | 'unresolved';
export type SourceLockPriority = 'stale_or_unresolved' | 'single_source' | 'allowed_sources' | 'auto';

export interface CoordinateFrame {
  id: string;
  canvas: [number, number];
  origin: [number, number];
  units: 'native_output_pixels';
}

export interface SourceBinding {
  source_id: number;
  path: string;
  sha256: string;
  station_id: number;
  capture_group_id: number;
  source_to_world_sha256: string;
}

export interface WorldRegion {
  polygon_world: Array<[number, number]>;
  bbox: [number, number, number, number];
}

export interface SourceLock {
  lock_id: string;
  status: SourceLockStatus;
  region: WorldRegion;
  requested_sources: number[];
  policy: SourceLockPolicy;
  eligible_tiles: number;
  selection_digest: string;
  created_event: number;
  stale_reason: string | null;
}

export interface ManualReviewRegion {
  region_id: string;
  polygon_world: Array<[number, number]>;
  reason: string;
  excluded_from_quality_gate: true;
  created_event: number;
}

export type SourceLockEventKind =
  | 'add_source_lock'
  | 'remove_source_lock'
  | 'add_manual_review_region'
  | 'remove_manual_review_region'
  | 'mark_lock_stale'
  | 'undo'
  | 'redo';

export interface SourceLockEvent {
  event_id: number;
  kind: SourceLockEventKind;
  inverse_kind: SourceLockEventKind;
  before_digest: string;
  after_digest: string;
  coordinate_frame_id: string;
  target_id: string;
  reverted_event_id?: number;
  payload?: SourceLock | ManualReviewRegion | { stale_reason: string } | null;
}

export interface SourceLockDocument {
  schema: typeof SOURCE_LOCK_SCHEMA_VERSION;
  run_id: string;
  input_manifest_sha256: string;
  coordinate_frame: CoordinateFrame;
  source_bindings: SourceBinding[];
  locks: SourceLock[];
  manual_review_regions: ManualReviewRegion[];
  history: {
    head: number;
    events: SourceLockEvent[];
  };
}

export interface RegionCoverageEvidence {
  coordinate_frame_id: string;
  input_manifest_sha256: string;
  eligible_tiles: number;
  covered_tiles: number;
  has_no_owner_or_transparent: boolean;
  footprint_and_inverse_valid: boolean;
  integrity_errors: string[];
  coverage_owner_tile_digest: string;
}

export interface LockEligibilityResult {
  eligible: boolean;
  reasons: string[];
}

export interface SessionContext {
  base_run_id: string;
  coordinate_frame_id: string;
  expected_head: number;
}

export type ExportBindingStatus = 'valid' | 'stale_export' | 'binding_mismatch' | 'has_stale_or_unresolved_locks';

export interface ExportBindingVerification {
  status: ExportBindingStatus;
  canPublishAsConfirmed: boolean;
  activeLocks: number;
  staleOrUnresolvedLocks: number;
  manualReviewRegions: number;
  sidecarSha256: string;
  reasons: string[];
}

export interface NormalizedBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WorldBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SidecarExportSummary {
  activeLockCount: number;
  activeLockAreaPx: number;
  staleOrUnresolvedCount: number;
  staleOrUnresolvedAreaPx: number;
  manualReviewCount: number;
  manualReviewAreaPx: number;
  requiresStaleConfirmation: boolean;
  sidecarSha256: string;
}

export interface SidecarImportReconciliation {
  doc: SourceLockDocument;
  newlyStaleCount: number;
  mismatchReasons: string[];
}

export interface ManualReviewQualityGateAccounting {
  all_tiles: number;
  auto_eligible_tiles: number;
  excluded_tiles: Array<{ tile_id: string; region_id: string; reason: string }>;
  failed_all_tiles: number;
  failed_auto_eligible_tiles: number;
  verdict: 'pass' | 'fail' | 'insufficient_evidence';
}

// Pure RFC 6234 SHA-256 so digest calculation is synchronous and identical
// across browser WebViews and Node unit tests with zero platform divergence.
const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
  0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
  0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
  0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
  0xc67178f2,
]);

export function sha256Hex(text: string): string {
  const bytes = new TextEncoder().encode(text);
  const bitLen = bytes.length * 8;
  const paddedLen = (((bytes.length + 9 + 63) >> 6) << 6) >>> 0;
  const padded = new Uint8Array(paddedLen);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 8, Math.floor(bitLen / 0x100000000), false);
  view.setUint32(paddedLen - 4, bitLen >>> 0, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i += 1) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i += 1) {
      const s0 = ((w[i - 15] >>> 7) | (w[i - 15] << 25)) ^ ((w[i - 15] >>> 18) | (w[i - 15] << 14)) ^ (w[i - 15] >>> 3);
      const s1 = ((w[i - 2] >>> 17) | (w[i - 2] << 15)) ^ ((w[i - 2] >>> 19) | (w[i - 2] << 13)) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i += 1) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7].map((word) => word.toString(16).padStart(8, '0')).join('');
}

const canonicalJson = (value: unknown): string => {
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalJson(item)).join(',')}]`;
  }
  if (value !== null && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    return `{${entries.map(([key, v]) => `${JSON.stringify(key)}:${canonicalJson(v)}`).join(',')}}`;
  }
  return JSON.stringify(value);
};

export function computeWorldRegion(polygon: Array<[number, number]>): WorldRegion {
  if (polygon.length < 3) {
    throw new Error('World region polygon requires at least 3 vertices.');
  }
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of polygon) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      throw new Error('World region polygon coordinates must be finite numbers.');
    }
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return {
    polygon_world: polygon.map(([x, y]) => [x, y]),
    bbox: [minX, minY, Math.max(0, maxX - minX), Math.max(0, maxY - minY)],
  };
}

export function computeRegionPixelArea(polygon: Array<[number, number]>): number {
  if (polygon.length < 3) return 0;
  let areaTwice = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const [x0, y0] = polygon[i];
    const [x1, y1] = polygon[(i + 1) % polygon.length];
    areaTwice += x0 * y1 - x1 * y0;
  }
  return Math.round(Math.abs(areaTwice) / 2);
}

const clamp01 = (value: number): number => Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));

export function normalizedBoxToWorldBox(box: NormalizedBox, frame: CoordinateFrame): WorldBox {
  const x0 = clamp01(Math.min(box.x, box.x + box.width));
  const y0 = clamp01(Math.min(box.y, box.y + box.height));
  const x1 = clamp01(Math.max(box.x, box.x + box.width));
  const y1 = clamp01(Math.max(box.y, box.y + box.height));
  const canvasW = Math.max(1, frame.canvas[0]);
  const canvasH = Math.max(1, frame.canvas[1]);
  const worldX = Math.round(frame.origin[0] + x0 * canvasW);
  const worldY = Math.round(frame.origin[1] + y0 * canvasH);
  const worldW = Math.max(1, Math.round((x1 - x0) * canvasW));
  const worldH = Math.max(1, Math.round((y1 - y0) * canvasH));
  return { x: worldX, y: worldY, width: worldW, height: worldH };
}

export function worldBoxToPolygon(box: WorldBox): Array<[number, number]> {
  const x = Math.round(box.x);
  const y = Math.round(box.y);
  const w = Math.max(1, Math.round(box.width));
  const h = Math.max(1, Math.round(box.height));
  return [
    [x, y],
    [x + w, y],
    [x + w, y + h],
    [x, y + h],
  ];
}

export function worldPolygonToNormalizedBox(polygon: Array<[number, number]>, frame: CoordinateFrame): NormalizedBox {
  const { bbox } = computeWorldRegion(polygon);
  const canvasW = Math.max(1, frame.canvas[0]);
  const canvasH = Math.max(1, frame.canvas[1]);
  const x = clamp01((bbox[0] - frame.origin[0]) / canvasW);
  const y = clamp01((bbox[1] - frame.origin[1]) / canvasH);
  const right = clamp01((bbox[0] + bbox[2] - frame.origin[0]) / canvasW);
  const bottom = clamp01((bbox[1] + bbox[3] - frame.origin[1]) / canvasH);
  return {
    x,
    y,
    width: Math.max(0.002, right - x),
    height: Math.max(0.002, bottom - y),
  };
}

export function createSourceLockDocument(params: {
  runId: string;
  inputManifestSha256: string;
  coordinateFrame: CoordinateFrame;
  sourceBindings: SourceBinding[];
}): SourceLockDocument {
  const seenIds = new Set<number>();
  for (const binding of params.sourceBindings) {
    if (!Number.isInteger(binding.source_id) || binding.source_id <= 0) {
      throw new Error(`Invalid source_id in manifest binding: ${binding.source_id}`);
    }
    if (seenIds.has(binding.source_id)) {
      throw new Error(`Duplicate source_id in manifest bindings: ${binding.source_id}`);
    }
    seenIds.add(binding.source_id);
  }
  return {
    schema: SOURCE_LOCK_SCHEMA_VERSION,
    run_id: params.runId,
    input_manifest_sha256: params.inputManifestSha256,
    coordinate_frame: {
      id: params.coordinateFrame.id,
      canvas: [params.coordinateFrame.canvas[0], params.coordinateFrame.canvas[1]],
      origin: [params.coordinateFrame.origin[0], params.coordinateFrame.origin[1]],
      units: 'native_output_pixels',
    },
    source_bindings: [...params.sourceBindings].sort((left, right) => left.source_id - right.source_id),
    locks: [],
    manual_review_regions: [],
    history: { head: 0, events: [] },
  };
}

/**
 * Compute deterministic SHA-256 selection digest over sorted source IDs,
 * world polygon, Coverage/owner tile digest, source-to-world fingerprints,
 * and policy. RGB pixel data is intentionally excluded.
 */
export function computeSelectionDigest(params: {
  requestedSources: number[];
  region: WorldRegion;
  coverageOwnerTileDigest: string;
  sourceBindings: SourceBinding[];
  policy: SourceLockPolicy;
  coordinateFrameId: string;
}): string {
  const sortedIds = [...new Set(params.requestedSources)].sort((a, b) => a - b);
  const byId = new Map(params.sourceBindings.map((binding) => [binding.source_id, binding]));
  const fingerprints = sortedIds.map((sourceId) => {
    const binding = byId.get(sourceId);
    return {
      source_id: sourceId,
      sha256: binding?.sha256 ?? '',
      source_to_world_sha256: binding?.source_to_world_sha256 ?? '',
    };
  });
  return sha256Hex(
    canonicalJson({
      coordinate_frame_id: params.coordinateFrameId,
      coverage_owner_tile_digest: params.coverageOwnerTileDigest,
      policy: params.policy,
      polygon_world: params.region.polygon_world,
      requested_sources: sortedIds,
      source_fingerprints: fingerprints,
    }),
  );
}

export function computeStateDigest(doc: Pick<SourceLockDocument, 'locks' | 'manual_review_regions'>): string {
  return sha256Hex(
    canonicalJson({
      locks: doc.locks,
      manual_review_regions: doc.manual_review_regions,
    }),
  );
}

export function computeSidecarDigest(doc: SourceLockDocument): string {
  return sha256Hex(canonicalJson(doc));
}

export function validateLockEligibility(
  doc: SourceLockDocument,
  requestedSourceIds: number[],
  policy: SourceLockPolicy,
  evidence: RegionCoverageEvidence,
): LockEligibilityResult {
  const reasons: string[] = [];
  const uniqueSources = [...new Set(requestedSourceIds)].sort((a, b) => a - b);
  if (uniqueSources.length === 0) {
    reasons.push('no_requested_sources');
  }
  if (policy === 'single_source' && uniqueSources.length !== 1) {
    reasons.push('single_source_policy_requires_exactly_one_source');
  }
  const byId = new Map(doc.source_bindings.map((binding) => [binding.source_id, binding]));
  for (const sourceId of uniqueSources) {
    const binding = byId.get(sourceId);
    if (!binding) {
      reasons.push(`unregistered_source_id:${sourceId}`);
      continue;
    }
    if (!binding.path.trim()) reasons.push(`missing_source_path:${sourceId}`);
    if (!binding.sha256.trim()) reasons.push(`missing_source_sha256:${sourceId}`);
    if (!binding.source_to_world_sha256.trim()) reasons.push(`missing_source_to_world:${sourceId}`);
  }
  if (evidence.coordinate_frame_id !== doc.coordinate_frame.id) {
    reasons.push('coordinate_frame_mismatch');
  }
  if (evidence.input_manifest_sha256 !== doc.input_manifest_sha256) {
    reasons.push('input_manifest_mismatch');
  }
  if (evidence.eligible_tiles <= 0 || evidence.covered_tiles < evidence.eligible_tiles) {
    reasons.push('incomplete_tile_coverage');
  }
  if (evidence.has_no_owner_or_transparent) {
    reasons.push('uncovered_or_no_owner_pixels');
  }
  if (!evidence.footprint_and_inverse_valid) {
    reasons.push('invalid_source_footprint_or_inverse');
  }
  if (!evidence.coverage_owner_tile_digest.trim()) {
    reasons.push('missing_coverage_owner_tile_digest');
  }
  if (evidence.integrity_errors.length > 0) {
    reasons.push(...evidence.integrity_errors.map((err) => `integrity_error:${err}`));
  }
  return { eligible: reasons.length === 0, reasons };
}

const assertSessionMatches = (doc: SourceLockDocument, session: SessionContext): void => {
  if (session.base_run_id !== doc.run_id) {
    throw new Error(`Session run_id mismatch: expected ${doc.run_id}, received ${session.base_run_id}.`);
  }
  if (session.coordinate_frame_id !== doc.coordinate_frame.id) {
    throw new Error(
      `Coordinate frame mismatch: expected ${doc.coordinate_frame.id}, received ${session.coordinate_frame_id}.`,
    );
  }
  if (session.expected_head !== doc.history.head) {
    throw new Error(
      `Concurrent session conflict: expected head event ${session.expected_head}, current head is ${doc.history.head}. Reload before committing.`,
    );
  }
};

export function commitAddSourceLock(
  doc: SourceLockDocument,
  session: SessionContext,
  params: {
    lockId?: string;
    polygonWorld: Array<[number, number]>;
    requestedSources: number[];
    policy: SourceLockPolicy;
    evidence: RegionCoverageEvidence;
  },
): SourceLockDocument {
  assertSessionMatches(doc, session);
  const eligibility = validateLockEligibility(doc, params.requestedSources, params.policy, params.evidence);
  if (!eligibility.eligible) {
    throw new Error(`Source lock rejected (${eligibility.reasons.join(', ')}).`);
  }
  const region = computeWorldRegion(params.polygonWorld);
  const sortedSources = [...new Set(params.requestedSources)].sort((a, b) => a - b);
  const nextEventId = doc.history.head + 1;
  const lockId = params.lockId ?? `lock-${String(nextEventId).padStart(4, '0')}`;
  const lock: SourceLock = {
    lock_id: lockId,
    status: 'active',
    region,
    requested_sources: sortedSources,
    policy: params.policy,
    eligible_tiles: params.evidence.eligible_tiles,
    selection_digest: computeSelectionDigest({
      requestedSources: sortedSources,
      region,
      coverageOwnerTileDigest: params.evidence.coverage_owner_tile_digest,
      sourceBindings: doc.source_bindings,
      policy: params.policy,
      coordinateFrameId: doc.coordinate_frame.id,
    }),
    created_event: nextEventId,
    stale_reason: null,
  };
  const beforeDigest = computeStateDigest(doc);
  const nextLocks = [...doc.locks, lock];
  const afterDigest = computeStateDigest({ locks: nextLocks, manual_review_regions: doc.manual_review_regions });
  const event: SourceLockEvent = {
    event_id: nextEventId,
    kind: 'add_source_lock',
    inverse_kind: 'remove_source_lock',
    before_digest: beforeDigest,
    after_digest: afterDigest,
    coordinate_frame_id: doc.coordinate_frame.id,
    target_id: lockId,
    payload: lock,
  };
  return {
    ...doc,
    locks: nextLocks,
    history: {
      head: nextEventId,
      events: [...doc.history.events, event],
    },
  };
}

export function commitAddManualReviewRegion(
  doc: SourceLockDocument,
  session: SessionContext,
  params: {
    regionId?: string;
    polygonWorld: Array<[number, number]>;
    reason: string;
  },
): SourceLockDocument {
  assertSessionMatches(doc, session);
  if (!params.reason.trim()) {
    throw new Error('Manual review region requires a non-empty reason.');
  }
  const region = computeWorldRegion(params.polygonWorld);
  const nextEventId = doc.history.head + 1;
  const regionId = params.regionId ?? `review-${String(nextEventId).padStart(4, '0')}`;
  const reviewRegion: ManualReviewRegion = {
    region_id: regionId,
    polygon_world: region.polygon_world,
    reason: params.reason.trim(),
    excluded_from_quality_gate: true,
    created_event: nextEventId,
  };
  const beforeDigest = computeStateDigest(doc);
  const nextRegions = [...doc.manual_review_regions, reviewRegion];
  const afterDigest = computeStateDigest({ locks: doc.locks, manual_review_regions: nextRegions });
  const event: SourceLockEvent = {
    event_id: nextEventId,
    kind: 'add_manual_review_region',
    inverse_kind: 'remove_manual_review_region',
    before_digest: beforeDigest,
    after_digest: afterDigest,
    coordinate_frame_id: doc.coordinate_frame.id,
    target_id: regionId,
    payload: reviewRegion,
  };
  return {
    ...doc,
    manual_review_regions: nextRegions,
    history: {
      head: nextEventId,
      events: [...doc.history.events, event],
    },
  };
}

export function commitRemoveSourceLock(
  doc: SourceLockDocument,
  session: SessionContext,
  lockId: string,
): SourceLockDocument {
  assertSessionMatches(doc, session);
  const existing = doc.locks.find((lock) => lock.lock_id === lockId);
  if (!existing) {
    throw new Error(`Source lock not found: ${lockId}`);
  }
  const nextEventId = doc.history.head + 1;
  const beforeDigest = computeStateDigest(doc);
  const nextLocks = doc.locks.filter((lock) => lock.lock_id !== lockId);
  const afterDigest = computeStateDigest({ locks: nextLocks, manual_review_regions: doc.manual_review_regions });
  const event: SourceLockEvent = {
    event_id: nextEventId,
    kind: 'remove_source_lock',
    inverse_kind: 'add_source_lock',
    before_digest: beforeDigest,
    after_digest: afterDigest,
    coordinate_frame_id: doc.coordinate_frame.id,
    target_id: lockId,
    payload: existing,
  };
  return {
    ...doc,
    locks: nextLocks,
    history: {
      head: nextEventId,
      events: [...doc.history.events, event],
    },
  };
}

export function commitRemoveManualReviewRegion(
  doc: SourceLockDocument,
  session: SessionContext,
  regionId: string,
): SourceLockDocument {
  assertSessionMatches(doc, session);
  const existing = doc.manual_review_regions.find((region) => region.region_id === regionId);
  if (!existing) {
    throw new Error(`Manual review region not found: ${regionId}`);
  }
  const nextEventId = doc.history.head + 1;
  const beforeDigest = computeStateDigest(doc);
  const nextRegions = doc.manual_review_regions.filter((region) => region.region_id !== regionId);
  const afterDigest = computeStateDigest({ locks: doc.locks, manual_review_regions: nextRegions });
  const event: SourceLockEvent = {
    event_id: nextEventId,
    kind: 'remove_manual_review_region',
    inverse_kind: 'add_manual_review_region',
    before_digest: beforeDigest,
    after_digest: afterDigest,
    coordinate_frame_id: doc.coordinate_frame.id,
    target_id: regionId,
    payload: existing,
  };
  return {
    ...doc,
    manual_review_regions: nextRegions,
    history: {
      head: nextEventId,
      events: [...doc.history.events, event],
    },
  };
}

const isUndoableCommand = (kind: SourceLockEventKind): boolean =>
  kind === 'add_source_lock' ||
  kind === 'remove_source_lock' ||
  kind === 'add_manual_review_region' ||
  kind === 'remove_manual_review_region';

const computeUndoRedoStacks = (
  events: SourceLockEvent[],
): { applied: SourceLockEvent[]; redoable: SourceLockEvent[] } => {
  const applied: SourceLockEvent[] = [];
  const redoable: SourceLockEvent[] = [];
  for (const event of events) {
    if (isUndoableCommand(event.kind)) {
      applied.push(event);
      redoable.length = 0;
    } else if (event.kind === 'undo') {
      const revertedId = event.reverted_event_id;
      if (applied.length > 0 && (revertedId === undefined || applied[applied.length - 1].event_id === revertedId)) {
        const popped = applied.pop()!;
        redoable.push(popped);
      } else if (revertedId !== undefined) {
        const idx = applied.findIndex((item) => item.event_id === revertedId);
        if (idx >= 0) {
          const [removed] = applied.splice(idx, 1);
          redoable.push(removed);
        }
      }
    } else if (event.kind === 'redo') {
      const reappliedId = event.reverted_event_id;
      if (
        redoable.length > 0 &&
        (reappliedId === undefined || redoable[redoable.length - 1].event_id === reappliedId)
      ) {
        const popped = redoable.pop()!;
        applied.push(popped);
      } else if (reappliedId !== undefined) {
        const idx = redoable.findIndex((item) => item.event_id === reappliedId);
        if (idx >= 0) {
          const [restored] = redoable.splice(idx, 1);
          applied.push(restored);
        }
      }
    }
  }
  return { applied, redoable };
};

export function canUndo(doc: SourceLockDocument): boolean {
  return computeUndoRedoStacks(doc.history.events).applied.length > 0;
}

export function canRedo(doc: SourceLockDocument): boolean {
  return computeUndoRedoStacks(doc.history.events).redoable.length > 0;
}

const applyEventState = (
  doc: SourceLockDocument,
  targetEvent: SourceLockEvent,
  direction: 'forward' | 'inverse',
): { locks: SourceLock[]; manual_review_regions: ManualReviewRegion[] } => {
  let nextLocks = doc.locks;
  let nextRegions = doc.manual_review_regions;
  const effectiveKind = direction === 'forward' ? targetEvent.kind : targetEvent.inverse_kind;
  if (effectiveKind === 'remove_source_lock') {
    nextLocks = doc.locks.filter((lock) => lock.lock_id !== targetEvent.target_id);
  } else if (effectiveKind === 'add_source_lock' && targetEvent.payload && 'lock_id' in targetEvent.payload) {
    nextLocks = [...doc.locks.filter((lock) => lock.lock_id !== targetEvent.target_id), targetEvent.payload];
  } else if (effectiveKind === 'remove_manual_review_region') {
    nextRegions = doc.manual_review_regions.filter((region) => region.region_id !== targetEvent.target_id);
  } else if (
    effectiveKind === 'add_manual_review_region' &&
    targetEvent.payload &&
    'region_id' in targetEvent.payload
  ) {
    nextRegions = [
      ...doc.manual_review_regions.filter((region) => region.region_id !== targetEvent.target_id),
      targetEvent.payload,
    ];
  }
  return { locks: nextLocks, manual_review_regions: nextRegions };
};

/**
 * Undo the latest committed command by appending an inverse event without
 * deleting prior history entries.
 */
export function commitUndo(doc: SourceLockDocument, session: SessionContext): SourceLockDocument {
  assertSessionMatches(doc, session);
  const { applied } = computeUndoRedoStacks(doc.history.events);
  const targetEvent = applied.length > 0 ? applied[applied.length - 1] : null;
  if (!targetEvent) {
    throw new Error('No committed command available to undo.');
  }

  const { locks: nextLocks, manual_review_regions: nextRegions } = applyEventState(doc, targetEvent, 'inverse');
  const nextEventId = doc.history.head + 1;
  const beforeDigest = computeStateDigest(doc);
  const afterDigest = computeStateDigest({ locks: nextLocks, manual_review_regions: nextRegions });
  const undoEvent: SourceLockEvent = {
    event_id: nextEventId,
    kind: 'undo',
    inverse_kind: targetEvent.kind,
    before_digest: beforeDigest,
    after_digest: afterDigest,
    coordinate_frame_id: doc.coordinate_frame.id,
    target_id: targetEvent.target_id,
    reverted_event_id: targetEvent.event_id,
    payload: targetEvent.payload ?? null,
  };
  return {
    ...doc,
    locks: nextLocks,
    manual_review_regions: nextRegions,
    history: {
      head: nextEventId,
      events: [...doc.history.events, undoEvent],
    },
  };
}

/**
 * Redo the most recently undone command by appending a monotonic redo event.
 * Any new non-undo/redo commit automatically clears the redo stack.
 */
export function commitRedo(doc: SourceLockDocument, session: SessionContext): SourceLockDocument {
  assertSessionMatches(doc, session);
  const { redoable } = computeUndoRedoStacks(doc.history.events);
  const targetEvent = redoable.length > 0 ? redoable[redoable.length - 1] : null;
  if (!targetEvent) {
    throw new Error('No undone command available to redo.');
  }

  const { locks: nextLocks, manual_review_regions: nextRegions } = applyEventState(doc, targetEvent, 'forward');
  const nextEventId = doc.history.head + 1;
  const beforeDigest = computeStateDigest(doc);
  const afterDigest = computeStateDigest({ locks: nextLocks, manual_review_regions: nextRegions });
  const redoEvent: SourceLockEvent = {
    event_id: nextEventId,
    kind: 'redo',
    inverse_kind: targetEvent.inverse_kind,
    before_digest: beforeDigest,
    after_digest: afterDigest,
    coordinate_frame_id: doc.coordinate_frame.id,
    target_id: targetEvent.target_id,
    reverted_event_id: targetEvent.event_id,
    payload: targetEvent.payload ?? null,
  };
  return {
    ...doc,
    locks: nextLocks,
    manual_review_regions: nextRegions,
    history: {
      head: nextEventId,
      events: [...doc.history.events, redoEvent],
    },
  };
}

/**
 * Mark affected locks `stale` whenever manifest SHA, coordinate frame
 * (canvas/origin/id), or referenced source SHA / source-to-world transform changes.
 */
export function reconcileDocumentStaleness(
  doc: SourceLockDocument,
  current: {
    inputManifestSha256: string;
    coordinateFrame: CoordinateFrame;
    sourceBindings: SourceBinding[];
  },
): SourceLockDocument {
  const frameChanged =
    current.coordinateFrame.id !== doc.coordinate_frame.id ||
    current.coordinateFrame.canvas[0] !== doc.coordinate_frame.canvas[0] ||
    current.coordinateFrame.canvas[1] !== doc.coordinate_frame.canvas[1] ||
    current.coordinateFrame.origin[0] !== doc.coordinate_frame.origin[0] ||
    current.coordinateFrame.origin[1] !== doc.coordinate_frame.origin[1];
  const manifestChanged = current.inputManifestSha256 !== doc.input_manifest_sha256;
  const currentById = new Map(current.sourceBindings.map((binding) => [binding.source_id, binding]));
  const priorById = new Map(doc.source_bindings.map((binding) => [binding.source_id, binding]));

  let changed = false;
  const nextLocks = doc.locks.map((lock) => {
    if (lock.status !== 'active') return lock;
    let staleReason: string | null = null;
    if (manifestChanged) {
      staleReason = 'input_manifest_changed';
    } else if (frameChanged) {
      staleReason = 'coordinate_frame_or_crop_changed';
    } else {
      for (const sourceId of lock.requested_sources) {
        const before = priorById.get(sourceId);
        const after = currentById.get(sourceId);
        if (!after) {
          staleReason = `source_removed:${sourceId}`;
          break;
        }
        if (!before || before.sha256 !== after.sha256) {
          staleReason = `source_digest_changed:${sourceId}`;
          break;
        }
        if (before.source_to_world_sha256 !== after.source_to_world_sha256) {
          staleReason = `source_transform_changed:${sourceId}`;
          break;
        }
      }
    }
    if (!staleReason) return lock;
    changed = true;
    return { ...lock, status: 'stale' as const, stale_reason: staleReason };
  });

  if (!changed) return doc;
  return { ...doc, locks: nextLocks };
}

/**
 * Precedence defined by Task 9:
 * `stale/unresolved` > `single_source` > `allowed_sources` > `auto`.
 */
export function resolveLockPriority(locksCoveringPoint: SourceLock[]): SourceLockPriority {
  if (locksCoveringPoint.some((lock) => lock.status === 'stale' || lock.status === 'unresolved')) {
    return 'stale_or_unresolved';
  }
  if (locksCoveringPoint.some((lock) => lock.status === 'active' && lock.policy === 'single_source')) {
    return 'single_source';
  }
  if (locksCoveringPoint.some((lock) => lock.status === 'active' && lock.policy === 'allowed_sources')) {
    return 'allowed_sources';
  }
  return 'auto';
}

export function verifyExportBinding(params: {
  doc: SourceLockDocument;
  currentTiffSha256: string;
  recordedTiffSha256: string;
  recordedSidecarSha256: string;
}): ExportBindingVerification {
  const sidecarSha256 = computeSidecarDigest(params.doc);
  const activeLocks = params.doc.locks.filter((lock) => lock.status === 'active').length;
  const staleOrUnresolvedLocks = params.doc.locks.filter((lock) => lock.status !== 'active').length;
  const manualReviewRegions = params.doc.manual_review_regions.length;
  const reasons: string[] = [];

  if (staleOrUnresolvedLocks > 0) {
    reasons.push(`${staleOrUnresolvedLocks} stale or unresolved lock(s) present`);
  }
  if (sidecarSha256 !== params.recordedSidecarSha256) {
    reasons.push('sidecar changed after TIFF was rendered (stale_export)');
  }
  if (params.currentTiffSha256 !== params.recordedTiffSha256) {
    reasons.push('TIFF digest does not match recorded sidecar binding (binding_mismatch)');
  }

  let status: ExportBindingStatus = 'valid';
  if (params.currentTiffSha256 !== params.recordedTiffSha256) {
    status = 'binding_mismatch';
  } else if (sidecarSha256 !== params.recordedSidecarSha256) {
    status = 'stale_export';
  } else if (staleOrUnresolvedLocks > 0) {
    status = 'has_stale_or_unresolved_locks';
  }

  return {
    status,
    canPublishAsConfirmed: status === 'valid',
    activeLocks,
    staleOrUnresolvedLocks,
    manualReviewRegions,
    sidecarSha256,
    reasons,
  };
}

/**
 * Automatic Quality Gate accounting with manual review regions excluded.
 * Manual regions never erase failures, and insufficient remaining samples
 * must fail closed as `insufficient_evidence`.
 */
export function computeManualReviewQualityGateAccounting(params: {
  allTiles: Array<{ tile_id: string; failed: boolean; manual_region_id?: string | null }>;
  doc: SourceLockDocument;
  minimumAutoSamples: number;
}): ManualReviewQualityGateAccounting {
  const regionById = new Map(params.doc.manual_review_regions.map((region) => [region.region_id, region]));
  const excludedTiles: Array<{ tile_id: string; region_id: string; reason: string }> = [];
  let autoEligibleTiles = 0;
  let failedAllTiles = 0;
  let failedAutoEligibleTiles = 0;

  for (const tile of params.allTiles) {
    if (tile.failed) failedAllTiles += 1;
    const matchedRegion = tile.manual_region_id ? regionById.get(tile.manual_region_id) : undefined;
    if (matchedRegion && matchedRegion.excluded_from_quality_gate) {
      excludedTiles.push({
        tile_id: tile.tile_id,
        region_id: matchedRegion.region_id,
        reason: matchedRegion.reason,
      });
    } else {
      autoEligibleTiles += 1;
      if (tile.failed) failedAutoEligibleTiles += 1;
    }
  }

  const verdict: 'pass' | 'fail' | 'insufficient_evidence' =
    failedAutoEligibleTiles > 0
      ? 'fail'
      : autoEligibleTiles < params.minimumAutoSamples
        ? 'insufficient_evidence'
        : 'pass';

  return {
    all_tiles: params.allTiles.length,
    auto_eligible_tiles: autoEligibleTiles,
    excluded_tiles: excludedTiles,
    failed_all_tiles: failedAllTiles,
    failed_auto_eligible_tiles: failedAutoEligibleTiles,
    verdict,
  };
}

export function serializeSidecarFiles(doc: SourceLockDocument): {
  sidecarJson: string;
  eventsJsonl: string;
} {
  return {
    sidecarJson: `${JSON.stringify(doc, null, 2)}\n`,
    eventsJsonl:
      doc.history.events.map((event) => JSON.stringify(event)).join('\n') + (doc.history.events.length ? '\n' : ''),
  };
}

const isFinitePair = (value: unknown): value is [number, number] =>
  Array.isArray(value) &&
  value.length === 2 &&
  typeof value[0] === 'number' &&
  Number.isFinite(value[0]) &&
  typeof value[1] === 'number' &&
  Number.isFinite(value[1]);

const isValidPolygon = (value: unknown): value is Array<[number, number]> =>
  Array.isArray(value) && value.length >= 3 && value.every(isFinitePair);

export function parseSourceLockDocument(rawJson: string): SourceLockDocument | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  const doc = parsed as Record<string, unknown>;
  if (doc.schema !== SOURCE_LOCK_SCHEMA_VERSION) return null;
  if (typeof doc.run_id !== 'string' || !doc.run_id.trim()) return null;
  if (typeof doc.input_manifest_sha256 !== 'string' || !doc.input_manifest_sha256.trim()) return null;

  const frame = doc.coordinate_frame as Record<string, unknown> | undefined;
  if (
    !frame ||
    typeof frame !== 'object' ||
    typeof frame.id !== 'string' ||
    !frame.id.trim() ||
    !isFinitePair(frame.canvas) ||
    frame.canvas[0] <= 0 ||
    frame.canvas[1] <= 0 ||
    !isFinitePair(frame.origin) ||
    frame.units !== 'native_output_pixels'
  ) {
    return null;
  }

  if (!Array.isArray(doc.source_bindings)) return null;
  const seenSourceIds = new Set<number>();
  const bindings: SourceBinding[] = [];
  for (const entry of doc.source_bindings) {
    if (!entry || typeof entry !== 'object') return null;
    const item = entry as Record<string, unknown>;
    if (
      typeof item.source_id !== 'number' ||
      !Number.isInteger(item.source_id) ||
      item.source_id <= 0 ||
      seenSourceIds.has(item.source_id) ||
      typeof item.path !== 'string' ||
      typeof item.sha256 !== 'string' ||
      typeof item.station_id !== 'number' ||
      typeof item.capture_group_id !== 'number' ||
      typeof item.source_to_world_sha256 !== 'string'
    ) {
      return null;
    }
    seenSourceIds.add(item.source_id);
    bindings.push({
      source_id: item.source_id,
      path: item.path,
      sha256: item.sha256,
      station_id: item.station_id,
      capture_group_id: item.capture_group_id,
      source_to_world_sha256: item.source_to_world_sha256,
    });
  }

  if (!Array.isArray(doc.locks) || !Array.isArray(doc.manual_review_regions)) return null;
  const locks: SourceLock[] = [];
  for (const entry of doc.locks) {
    if (!entry || typeof entry !== 'object') return null;
    const item = entry as Record<string, unknown>;
    const region = item.region as Record<string, unknown> | undefined;
    if (
      typeof item.lock_id !== 'string' ||
      !item.lock_id.trim() ||
      (item.status !== 'active' && item.status !== 'stale' && item.status !== 'unresolved') ||
      !region ||
      !isValidPolygon(region.polygon_world) ||
      !Array.isArray(item.requested_sources) ||
      !item.requested_sources.every((id) => typeof id === 'number' && Number.isInteger(id) && id > 0) ||
      (item.policy !== 'single_source' && item.policy !== 'allowed_sources') ||
      typeof item.eligible_tiles !== 'number' ||
      typeof item.selection_digest !== 'string' ||
      typeof item.created_event !== 'number' ||
      (item.stale_reason !== null && typeof item.stale_reason !== 'string')
    ) {
      return null;
    }
    locks.push({
      lock_id: item.lock_id,
      status: item.status,
      region: computeWorldRegion(region.polygon_world),
      requested_sources: [...new Set(item.requested_sources as number[])].sort((a, b) => a - b),
      policy: item.policy,
      eligible_tiles: item.eligible_tiles,
      selection_digest: item.selection_digest,
      created_event: item.created_event,
      stale_reason: item.stale_reason,
    });
  }

  const manualReviewRegions: ManualReviewRegion[] = [];
  for (const entry of doc.manual_review_regions) {
    if (!entry || typeof entry !== 'object') return null;
    const item = entry as Record<string, unknown>;
    if (
      typeof item.region_id !== 'string' ||
      !item.region_id.trim() ||
      !isValidPolygon(item.polygon_world) ||
      typeof item.reason !== 'string' ||
      !item.reason.trim() ||
      item.excluded_from_quality_gate !== true ||
      typeof item.created_event !== 'number'
    ) {
      return null;
    }
    manualReviewRegions.push({
      region_id: item.region_id,
      polygon_world: item.polygon_world.map(([x, y]) => [x, y]),
      reason: item.reason,
      excluded_from_quality_gate: true,
      created_event: item.created_event,
    });
  }

  const history = doc.history as Record<string, unknown> | undefined;
  if (
    !history ||
    typeof history !== 'object' ||
    typeof history.head !== 'number' ||
    !Number.isInteger(history.head) ||
    history.head < 0 ||
    !Array.isArray(history.events)
  ) {
    return null;
  }
  let prevEventId = 0;
  for (const eventEntry of history.events) {
    if (!eventEntry || typeof eventEntry !== 'object') return null;
    const ev = eventEntry as Record<string, unknown>;
    if (typeof ev.event_id !== 'number' || !Number.isInteger(ev.event_id) || ev.event_id <= prevEventId) {
      return null;
    }
    prevEventId = ev.event_id;
  }
  if (prevEventId > history.head) return null;

  return {
    schema: SOURCE_LOCK_SCHEMA_VERSION,
    run_id: doc.run_id,
    input_manifest_sha256: doc.input_manifest_sha256,
    coordinate_frame: {
      id: frame.id,
      canvas: [frame.canvas[0], frame.canvas[1]],
      origin: [frame.origin[0], frame.origin[1]],
      units: 'native_output_pixels',
    },
    source_bindings: bindings.sort((left, right) => left.source_id - right.source_id),
    locks,
    manual_review_regions: manualReviewRegions,
    history: {
      head: history.head,
      events: history.events as SourceLockEvent[],
    },
  };
}

/**
 * Parse an imported `source-locks.json` sidecar and reconcile every active lock
 * against the currently loaded run's identity, manifest, coordinate frame, and
 * source bindings. Any mismatch marks affected locks `stale` so they can never
 * silently affect a different render or publish as confirmed.
 */
export function importAndReconcileSidecar(
  rawJson: string,
  current: {
    runId: string;
    inputManifestSha256: string;
    coordinateFrame: CoordinateFrame;
    sourceBindings: SourceBinding[];
  },
): SidecarImportReconciliation | null {
  const parsed = parseSourceLockDocument(rawJson);
  if (!parsed) return null;

  const activeBefore = parsed.locks.filter((lock) => lock.status === 'active').length;
  const mismatchReasons: string[] = [];

  let reconciled = parsed;
  if (parsed.run_id !== current.runId) {
    mismatchReasons.push(`run_id_changed:${parsed.run_id}->${current.runId}`);
    reconciled = {
      ...parsed,
      locks: parsed.locks.map((lock) =>
        lock.status === 'active' ? { ...lock, status: 'stale' as const, stale_reason: 'run_id_changed' } : lock,
      ),
    };
  } else {
    reconciled = reconcileDocumentStaleness(parsed, {
      inputManifestSha256: current.inputManifestSha256,
      coordinateFrame: current.coordinateFrame,
      sourceBindings: current.sourceBindings,
    });
    for (const lock of reconciled.locks) {
      if (lock.status !== 'active' && lock.stale_reason && !mismatchReasons.includes(lock.stale_reason)) {
        mismatchReasons.push(lock.stale_reason);
      }
    }
  }

  const activeAfter = reconciled.locks.filter((lock) => lock.status === 'active').length;
  return {
    doc: reconciled,
    newlyStaleCount: Math.max(0, activeBefore - activeAfter),
    mismatchReasons,
  };
}

export function summarizeSidecarForExport(doc: SourceLockDocument): SidecarExportSummary {
  let activeLockCount = 0;
  let activeLockAreaPx = 0;
  let staleOrUnresolvedCount = 0;
  let staleOrUnresolvedAreaPx = 0;
  for (const lock of doc.locks) {
    const area = computeRegionPixelArea(lock.region.polygon_world);
    if (lock.status === 'active') {
      activeLockCount += 1;
      activeLockAreaPx += area;
    } else {
      staleOrUnresolvedCount += 1;
      staleOrUnresolvedAreaPx += area;
    }
  }
  let manualReviewAreaPx = 0;
  for (const region of doc.manual_review_regions) {
    manualReviewAreaPx += computeRegionPixelArea(region.polygon_world);
  }
  return {
    activeLockCount,
    activeLockAreaPx,
    staleOrUnresolvedCount,
    staleOrUnresolvedAreaPx,
    manualReviewCount: doc.manual_review_regions.length,
    manualReviewAreaPx,
    requiresStaleConfirmation: staleOrUnresolvedCount > 0,
    sidecarSha256: computeSidecarDigest(doc),
  };
}
