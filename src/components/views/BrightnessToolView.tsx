import {
  examineNativeTexture,
  summarizeNativeTexture,
  type NativeTextureEvidence,
} from '../../utils/nativeSeamTexture';
import { examineNativeSeamProfile, summarizeNativeSeam } from '../../utils/nativeSeamProfile';
import { scanNativeSeam, resolveNativeSeamSequences } from '../../utils/nativeSeamPath';
import { detectWeakSeamCandidates } from '../../utils/weakSeamCandidates';
import { detectSeamCandidates, type SeamCandidate } from '../../utils/seamCandidates';
import { analyzeIllumination, applyIllumination, type IlluminationModel } from '../../utils/adaptiveBrightness';
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
} from 'react';
import {
  ArrowLeft,
  Eye,
  EyeOff,
  Grid2X2,
  ImagePlus,
  RefreshCcw,
  ScanSearch,
  Sun,
  Upload,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { save } from '@tauri-apps/plugin-dialog';
import { useTranslation } from 'react-i18next';
import ExportImageDialog from '../../features/export/ExportImageDialog';
import {
  EXPORT_DIALOG_FORMATS,
  buildBackendExportSettings,
  buildSuggestedExportPath,
  ensureExportPathExtension,
  exportDialogExtension,
} from '../../features/export/exportDialog';
import type { ExportDialogSettings, ExportDialogSource } from '../../features/export/exportDialog';
import { useSettingsStore } from '../../store/useSettingsStore';
import { Invokes } from '../ui/AppProperties';
import Button from '../ui/Button';
import { analyzeBrightnessRegions, applyBrightnessOffsets, type BrightnessGrid } from '../../utils/brightnessMap';

interface BrightnessToolViewProps {
  onBack(): void;
  initialImage?: {
    metadata?: Record<string, unknown> | null;
    name: string;
    path?: string;
    url: string;
  } | null;
}

interface SourceImage {
  metadata: Record<string, unknown> | null;
  name: string;
  path?: string;
  url: string;
  image: HTMLImageElement;
  owned: boolean;
}

const MAX_PREVIEW_EDGE = 1400;
const MAX_EXPORT_EDGE = 6000;
const GRID_ROWS = 4;
const GRID_COLUMNS = 6;
const MIN_PREVIEW_ZOOM = 1;
const MAX_PREVIEW_ZOOM = 6;

function formatBrightness(value: number) {
  return `${Math.round(value)}`;
}

function heatColor(value: number, minimum: number, maximum: number) {
  const span = Math.max(1, maximum - minimum);
  const normalized = Math.max(0, Math.min(1, (value - minimum) / span));
  const hue = 220 - normalized * 220;
  return `hsla(${hue}, 88%, 58%, 0.32)`;
}

export default function BrightnessToolView({ onBack, initialImage }: BrightnessToolViewProps) {
  const { t } = useTranslation();
  const translateRef = useRef(t);
  useEffect(() => {
    translateRef.current = t;
  }, [t]);
  const osPlatform = useSettingsStore((state) => state.osPlatform);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);
  const previewStageRef = useRef<HTMLDivElement>(null);
  const [source, setSource] = useState<SourceImage | null>(null);
  const [grid, setGrid] = useState<BrightnessGrid | null>(null);
  const [offsets, setOffsets] = useState<number[]>([]);
  const seamAnalysisId = useRef(0);
  const [seamsLoading, setSeamsLoading] = useState(false);
  const [seamProgress, setSeamProgress] = useState('');
  const [seams, setSeams] = useState<SeamCandidate[]>([]);
  const [seamsVisible, setSeamsVisible] = useState(false);
  const [seamsAnalyzed, setSeamsAnalyzed] = useState(false);
  const [selectedSeam, setSelectedSeam] = useState<number | null>(null);
  const [ignoredSeams, setIgnoredSeams] = useState<number[]>([]);
  const [illumination, setIllumination] = useState<IlluminationModel | null>(null);
  const [autoEnabled, setAutoEnabled] = useState(false);
  const [autoStrength, setAutoStrength] = useState(50);
  const [excludedRegions, setExcludedRegions] = useState<number[]>([]);
  const [selectedRegion, setSelectedRegion] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isBrightnessLayerVisible, setIsBrightnessLayerVisible] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [imageBounds, setImageBounds] = useState({ left: 0, top: 0, width: 0, height: 0 });
  const [previewZoom, setPreviewZoom] = useState(MIN_PREVIEW_ZOOM);
  const [previewPan, setPreviewPan] = useState({ x: 0, y: 0 });
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
  const activePointerRef = useRef<number | null>(null);
  const lastPointerPositionRef = useRef({ x: 0, y: 0 });

  const loadFile = useCallback(
    (file: File | undefined) => {
      if (!file || !file.type.startsWith('image/')) {
        setError(t('brightnessTool.invalidImage', { defaultValue: '请选择 PNG、JPEG 或其他图片文件。' }));
        return;
      }
      setError(null);
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.crossOrigin = 'anonymous';
      image.onload = () => setSource({ metadata: null, name: file.name, path: undefined, url, image, owned: true });
      image.onerror = () => {
        URL.revokeObjectURL(url);
        setError(t('brightnessTool.loadError', { defaultValue: '图片读取失败，请换一张图片重试。' }));
      };
      image.src = url;
    },
    [t],
  );

  useEffect(() => {
    if (!initialImage?.url) return;
    setError(null);
    const image = new Image();
    image.crossOrigin = 'anonymous';
    image.onload = () =>
      setSource({
        metadata: initialImage.metadata ?? null,
        name: initialImage.name,
        path: initialImage.path,
        url: initialImage.url,
        image,
        owned: false,
      });
    image.onerror = () =>
      setError(translateRef.current('brightnessTool.loadError', { defaultValue: '图片读取失败，请换一张图片重试。' }));
    image.src = initialImage.url;
  }, [initialImage?.metadata, initialImage?.name, initialImage?.path, initialImage?.url]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    loadFile(event.target.files?.[0]);
    event.target.value = '';
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    loadFile(event.dataTransfer.files?.[0]);
  };

  useEffect(() => {
    return () => {
      if (source?.owned && source.url) URL.revokeObjectURL(source.url);
    };
  }, [source?.owned, source?.url]);

  useEffect(() => {
    if (!source) {
      setGrid(null);
      setOffsets([]);
      return;
    }

    const scale = Math.min(1, MAX_PREVIEW_EDGE / Math.max(source.image.naturalWidth, source.image.naturalHeight));
    const width = Math.max(1, Math.round(source.image.naturalWidth * scale));
    const height = Math.max(1, Math.round(source.image.naturalHeight * scale));
    const canvas = sourceCanvasRef.current;
    if (!canvas) return;
    try {
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Canvas 2D context is unavailable.');
      context.clearRect(0, 0, width, height);
      context.drawImage(source.image, 0, 0, width, height);
      const nextGrid = analyzeBrightnessRegions(context.getImageData(0, 0, width, height), GRID_ROWS, GRID_COLUMNS);
      seamAnalysisId.current += 1;
      setSeamsLoading(false);
      setGrid(nextGrid);
      setSeams([]);
      setSeamsVisible(false);
      setSeamsAnalyzed(false);
      setSelectedSeam(null);
      setIgnoredSeams([]);
      setIllumination(analyzeIllumination(context.getImageData(0, 0, width, height)));
      setAutoEnabled(false);
      setExcludedRegions([]);
      setOffsets(new Array(nextGrid.regions.length).fill(0));
      setSelectedRegion(0);
    } catch (cause) {
      console.error('Failed to analyze brightness source:', cause);
      setGrid(null);
      setOffsets([]);
      setError(
        translateRef.current('brightnessTool.loadError', {
          defaultValue: '当前图片预览无法读取像素，请点击“更换图片”重新导入。',
        }),
      );
    }
  }, [source]);

  const protectedAreas = useMemo(
    () =>
      grid
        ? excludedRegions.map((index) => {
            const region = grid.regions[index];
            return {
              x: region.column / grid.columns,
              y: region.row / grid.rows,
              width: 1 / grid.columns,
              height: 1 / grid.rows,
            };
          })
        : [],
    [grid, excludedRegions],
  );
  // Manual edits are continuous and never inject grid-aligned seam steps.
  const manualGrid = useMemo(
    () => (grid ? { ...grid, verticalSeamDeltas: [], horizontalSeamDeltas: [] } : null),
    [grid],
  );

  const renderedImageData = useMemo(() => {
    if (!grid) return null;
    const canvas = sourceCanvasRef.current;
    const context = canvas?.getContext('2d', { willReadFrequently: true });
    if (!canvas || !context) return null;
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const automatic =
      autoEnabled && illumination
        ? applyIllumination(imageData, illumination, autoStrength / 100, protectedAreas)
        : imageData;
    return applyBrightnessOffsets(automatic, manualGrid ?? grid, offsets);
  }, [grid, offsets, autoEnabled, illumination, autoStrength, protectedAreas, manualGrid]);

  const renderedGrid = useMemo(
    () => (renderedImageData && grid ? analyzeBrightnessRegions(renderedImageData, grid.rows, grid.columns) : grid),
    [renderedImageData, grid],
  );

  useEffect(() => {
    const canvas = previewCanvasRef.current;
    if (!canvas || !renderedImageData) return;
    canvas.width = renderedImageData.width;
    canvas.height = renderedImageData.height;
    canvas.getContext('2d')?.putImageData(renderedImageData, 0, 0);
  }, [renderedImageData]);

  useEffect(() => {
    const stage = previewStageRef.current;
    if (!stage) return;
    const updateImageBounds = () => {
      const canvas = previewCanvasRef.current;
      const stageWidth = stage.clientWidth;
      const stageHeight = stage.clientHeight;
      const imageRatio = canvas && canvas.height > 0 ? canvas.width / canvas.height : 1;
      const width = Math.min(stageWidth, stageHeight * imageRatio);
      const height = width / imageRatio;
      setImageBounds({
        left: Math.max(0, (stageWidth - width) / 2),
        top: Math.max(0, (stageHeight - height) / 2),
        width,
        height,
      });
    };
    const observer = new ResizeObserver(updateImageBounds);
    observer.observe(stage);
    updateImageBounds();
    return () => observer.disconnect();
  }, [grid, renderedImageData]);

  const clampPreviewPan = useCallback(
    (scale: number, x: number, y: number) => {
      const stage = previewStageRef.current;
      if (!stage) return { x: 0, y: 0 };
      const maxX = Math.max(0, (imageBounds.width * scale - stage.clientWidth) / 2);
      const maxY = Math.max(0, (imageBounds.height * scale - stage.clientHeight) / 2);
      return {
        x: Math.max(-maxX, Math.min(maxX, x)),
        y: Math.max(-maxY, Math.min(maxY, y)),
      };
    },
    [imageBounds.height, imageBounds.width],
  );

  const applyPreviewZoom = useCallback(
    (nextScale: number, anchor?: { x: number; y: number }) => {
      const scale = Math.max(MIN_PREVIEW_ZOOM, Math.min(MAX_PREVIEW_ZOOM, nextScale));
      const currentScale = previewZoom;
      const ratio = scale / currentScale;
      let nextPan = { ...previewPan };
      if (anchor) {
        nextPan = {
          x: anchor.x - (anchor.x - previewPan.x) * ratio,
          y: anchor.y - (anchor.y - previewPan.y) * ratio,
        };
      } else if (scale <= MIN_PREVIEW_ZOOM) {
        nextPan = { x: 0, y: 0 };
      } else {
        nextPan = { x: previewPan.x * ratio, y: previewPan.y * ratio };
      }
      setPreviewZoom(scale);
      setPreviewPan(clampPreviewPan(scale, nextPan.x, nextPan.y));
    },
    [clampPreviewPan, previewPan, previewZoom],
  );

  const resetPreviewZoom = useCallback(() => {
    setPreviewZoom(MIN_PREVIEW_ZOOM);
    setPreviewPan({ x: 0, y: 0 });
  }, []);

  useEffect(() => {
    resetPreviewZoom();
  }, [resetPreviewZoom, source?.url]);

  const handlePreviewWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const delta = event.deltaY !== 0 ? event.deltaY : event.deltaX;
    const nextScale = Math.max(
      MIN_PREVIEW_ZOOM,
      Math.min(MAX_PREVIEW_ZOOM, previewZoom * Math.exp(Math.max(-0.5, Math.min(0.5, -delta * 0.002)))),
    );
    if (Math.abs(nextScale - previewZoom) < 0.0001) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    applyPreviewZoom(nextScale, {
      x: event.clientX - bounds.left - bounds.width / 2,
      y: event.clientY - bounds.top - bounds.height / 2,
    });
  };

  const handlePreviewPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0 || previewZoom <= MIN_PREVIEW_ZOOM) return;
    if (event.target instanceof Element && event.target.closest('button')) return;
    event.preventDefault();
    activePointerRef.current = event.pointerId;
    lastPointerPositionRef.current = { x: event.clientX, y: event.clientY };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePreviewPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (activePointerRef.current !== event.pointerId) return;
    event.preventDefault();
    const deltaX = event.clientX - lastPointerPositionRef.current.x;
    const deltaY = event.clientY - lastPointerPositionRef.current.y;
    lastPointerPositionRef.current = { x: event.clientX, y: event.clientY };
    setPreviewPan((current) => clampPreviewPan(previewZoom, current.x + deltaX, current.y + deltaY));
  };

  const handlePreviewPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (activePointerRef.current !== event.pointerId) return;
    activePointerRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handlePreviewKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      applyPreviewZoom(previewZoom * 1.25);
    } else if (event.key === '-') {
      event.preventDefault();
      applyPreviewZoom(previewZoom / 1.25);
    } else if (event.key === '0') {
      event.preventDefault();
      resetPreviewZoom();
    }
  };

  const updateOffset = (index: number, value: number) => {
    setSelectedRegion(index);
    setOffsets((current) => current.map((offset, currentIndex) => (currentIndex === index ? value : offset)));
  };

  useEffect(
    () => () => {
      seamAnalysisId.current += 1;
    },
    [],
  );

  const analyzeSeams = async () => {
    if (!source) return;
    const analysisId = ++seamAnalysisId.current;
    setSeamsLoading(true);
    setSeamProgress(t('brightnessTool.scanningSeams', { defaultValue: '扫描中' }));
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 2400 / Math.max(source.image.naturalWidth, source.image.naturalHeight));
    canvas.width = Math.max(1, Math.round(source.image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(source.image.naturalHeight * scale));
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) {
      setSeamsLoading(false);
      return;
    }
    try {
      context.drawImage(source.image, 0, 0, canvas.width, canvas.height);
      const analysisPixels = context.getImageData(0, 0, canvas.width, canvas.height);
      const candidates = detectSeamCandidates(analysisPixels);
      for (const proposal of detectWeakSeamCandidates(analysisPixels)) {
        if (proposal) candidates.push({ ...proposal, id: candidates.length });
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        if (analysisId !== seamAnalysisId.current) return;
      }
      const patchSize = 1280,
        patchCenter = patchSize / 2;
      const patch = document.createElement('canvas');
      patch.width = patchSize;
      patch.height = patchSize;
      const patchContext = patch.getContext('2d', { willReadFrequently: true });
      if (!patchContext) throw new Error('Canvas unavailable');
      const alternatives: SeamCandidate[] = [];
      for (const [candidateIndex, candidate] of candidates.entries()) {
        setSeamProgress(
          t('brightnessTool.verifyingSeams', {
            defaultValue: '复核 {{current}}/{{total}}',
            current: candidateIndex + 1,
            total: candidates.length,
          }),
        );
        const samples = [];
        const profileSamples = [];
        const scans = [];
        const patches: {
          pixels: ImageData;
          px: number;
          py: number;
          normal: [number, number];
          textureCache: Map<number, NativeTextureEvidence>;
        }[] = [];
        candidate.nativeLocations = [];
        const referenceNormal = candidate.normals[0];
        const ordered = candidate.points
          .map((_, index) => index)
          .sort(
            (a, b) =>
              -candidate.points[a][0] * source.image.naturalWidth * referenceNormal[1] +
              candidate.points[a][1] * source.image.naturalHeight * referenceNormal[0] -
              (-candidate.points[b][0] * source.image.naturalWidth * referenceNormal[1] +
                candidate.points[b][1] * source.image.naturalHeight * referenceNormal[0]),
          );
        const project = (index: number) =>
          (-candidate.points[index][0] * source.image.naturalWidth * referenceNormal[1] +
            candidate.points[index][1] * source.image.naturalHeight * referenceNormal[0]) /
          Math.hypot(...referenceNormal);
        const start = project(ordered[0]),
          end = project(ordered[ordered.length - 1]);
        const indices = [
          ...new Set(
            Array.from({ length: Math.min(8, ordered.length) }, (_, k) => {
              const target = start + ((end - start) * k) / Math.max(1, Math.min(8, ordered.length) - 1);
              return ordered.reduce(
                (best, index) => (Math.abs(project(index) - target) < Math.abs(project(best) - target) ? index : best),
                ordered[0],
              );
            }),
          ),
        ];
        const spacing =
          indices.length > 1
            ? Math.min(...indices.slice(1).map((index, i) => project(index) - project(indices[i])))
            : 0;
        const nativeSpan = Math.max(32, Math.min(960, spacing * 0.8));
        for (const index of indices) {
          const [x, y] = candidate.points[index];
          const px = Math.round(x * source.image.naturalWidth),
            py = Math.round(y * source.image.naturalHeight);
          patchContext.clearRect(0, 0, patchSize, patchSize);
          // Equal source/destination dimensions: no full-image canvas or resampling.
          patchContext.drawImage(
            source.image,
            px - patchCenter,
            py - patchCenter,
            patchSize,
            patchSize,
            0,
            0,
            patchSize,
            patchSize,
          );
          const pixels = patchContext.getImageData(0, 0, patchSize, patchSize);
          const normal = candidate.normals[index];
          const length = Math.hypot(...normal);
          patches.push({ pixels, px, py, normal, textureCache: new Map() });
          scans.push(scanNativeSeam(pixels, patchCenter, patchCenter, normal, nativeSpan));
          const profile = examineNativeSeamProfile(pixels, patchCenter, patchCenter, normal);
          const offset = profile.status === 'stable' ? profile.offset : 0;
          const dx = (offset * normal[0]) / length,
            dy = (offset * normal[1]) / length;
          const texture = examineNativeTexture(pixels, patchCenter + dx, patchCenter + dy, normal);
          patches[patches.length - 1].textureCache.set(offset, texture);
          samples.push(texture);
          profileSamples.push({ profile, textureSupported: texture.status === 'supported' });
          if (profile.status === 'stable' && texture.status === 'supported') {
            candidate.nativeLocations.push([
              (px + dx) / source.image.naturalWidth,
              (py + dy) / source.image.naturalHeight,
            ]);
          }
          await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
          if (analysisId !== seamAnalysisId.current) return;
        }
        candidate.nativeTexture = summarizeNativeTexture(samples);
        candidate.nativeProfile = summarizeNativeSeam(profileSamples);
        if (patches.every((p) => p.normal[0] === referenceNormal[0] && p.normal[1] === referenceNormal[1])) {
          const length = Math.hypot(...referenceNormal);
          const sequences = resolveNativeSeamSequences(
            scans,
            patches.map((p) => (p.px * referenceNormal[0] + p.py * referenceNormal[1]) / length),
          );
          for (const [sequenceIndex, sequence] of sequences.entries()) {
            const target =
              sequenceIndex === 0 ? candidate : { ...candidate, id: candidates.length + alternatives.length };
            target.nativeSequence = sequence;
            target.nativeLocations = [];
            target.nativePath = [];
            const textures = patches.map((p, i) => {
              const offset = sequence.offsets[i];
              const dx = (offset * p.normal[0]) / length,
                dy = (offset * p.normal[1]) / length;
              target.nativePath!.push([
                (p.px + dx) / source.image.naturalWidth,
                (p.py + dy) / source.image.naturalHeight,
              ]);
              let texture = p.textureCache.get(offset);
              if (!texture) {
                texture = examineNativeTexture(p.pixels, patchCenter + dx, patchCenter + dy, p.normal);
                p.textureCache.set(offset, texture);
              }
              if (texture.status === 'supported')
                target.nativeLocations!.push([
                  (p.px + dx) / source.image.naturalWidth,
                  (p.py + dy) / source.image.naturalHeight,
                ]);
              return texture;
            });
            target.nativeTexture = summarizeNativeTexture(textures);
            target.nativeProfile = {
              status: target.nativeTexture.status === 'supported' ? 'supported' : 'insufficient',
              samples: patches.length,
              supported: textures.filter((s) => s.status === 'supported').length,
              delta: sequence.delta,
            };
            if (sequenceIndex > 0) alternatives.push(target);
          }
        }
      }
      const verifiedCandidates: SeamCandidate[] = [];
      for (const candidate of [...candidates, ...alternatives].filter(
        (c) => c.proposal !== 'coherent' || c.nativeSequence?.status === 'stable',
      )) {
        const path = candidate.nativePath;
        const duplicate =
          path &&
          verifiedCandidates.some(
            (previous) =>
              previous.nativePath?.length === path.length &&
              path.every(
                (p, i) =>
                  Math.hypot(
                    (p[0] - previous.nativePath![i][0]) * source.image.naturalWidth,
                    (p[1] - previous.nativePath![i][1]) * source.image.naturalHeight,
                  ) < 6,
              ),
          );
        if (!duplicate) verifiedCandidates.push(candidate);
      }
      setSeams(verifiedCandidates);
      setSelectedSeam(verifiedCandidates[0]?.id ?? null);
      setIgnoredSeams([]);
      setSeamsAnalyzed(true);
      setSeamsVisible(true);
      setIsBrightnessLayerVisible(false);
    } catch {
      if (analysisId === seamAnalysisId.current)
        setError(t('brightnessTool.loadError', { defaultValue: '图片读取失败' }));
    } finally {
      if (analysisId === seamAnalysisId.current) setSeamsLoading(false);
    }
  };

  const normalizeImage = () => {
    if (!grid) return;
    setAutoEnabled(true);
  };

  const resetOffsets = () => {
    setOffsets((current) => current.map(() => 0));
    setAutoEnabled(false);
    setExcludedRegions([]);
    setAutoStrength(50);
  };

  const exportDimensions = useMemo(() => {
    if (!source) return { height: 1, width: 1 };
    const scale = Math.min(1, MAX_EXPORT_EDGE / Math.max(source.image.naturalWidth, source.image.naturalHeight));
    return {
      height: Math.max(1, Math.round(source.image.naturalHeight * scale)),
      width: Math.max(1, Math.round(source.image.naturalWidth * scale)),
    };
  }, [source]);

  const exportPreviewSrc = useMemo(() => {
    if (!renderedImageData) return source?.url || '';
    const canvas = document.createElement('canvas');
    canvas.width = renderedImageData.width;
    canvas.height = renderedImageData.height;
    canvas.getContext('2d')?.putImageData(renderedImageData, 0, 0);
    return canvas.toDataURL('image/png');
  }, [renderedImageData, source?.url]);

  const exportSource = useMemo<ExportDialogSource | null>(() => {
    if (!source || !grid || !exportPreviewSrc) return null;
    return {
      detailPreviewSrc: exportPreviewSrc,
      fileName: source.name,
      height: exportDimensions.height,
      previewSrc: exportPreviewSrc,
      width: exportDimensions.width,
    };
  }, [exportDimensions.height, exportDimensions.width, exportPreviewSrc, grid, source]);

  const handleExport = useCallback(
    async (settings: ExportDialogSettings): Promise<string | null> => {
      if (!source || !grid) return null;
      const format = EXPORT_DIALOG_FORMATS.find((candidate) => candidate.id === settings.format);
      if (!format) throw new Error(`Unsupported export format: ${settings.format}`);

      const suggestedName = buildSuggestedExportPath(source.path || source.name, '_brightness', settings.format);
      const selectedOutputPath =
        osPlatform === 'android'
          ? suggestedName
          : await save({
              defaultPath: suggestedName,
              filters: [{ extensions: format.extensions, name: format.label }],
              title: t('brightnessTool.download', { defaultValue: '导出调整结果' }),
            });
      if (!selectedOutputPath) return null;
      const outputPath = ensureExportPathExtension(selectedOutputPath, settings.format);
      const canvas = document.createElement('canvas');
      canvas.width = settings.resizeWidth;
      canvas.height = settings.resizeHeight;
      const context = canvas.getContext('2d', { willReadFrequently: true });
      if (!context) throw new Error('Canvas export is unavailable.');
      context.drawImage(source.image, 0, 0, canvas.width, canvas.height);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      const automatic =
        autoEnabled && illumination
          ? applyIllumination(pixels, illumination, autoStrength / 100, protectedAreas)
          : pixels;
      context.putImageData(applyBrightnessOffsets(automatic, manualGrid ?? grid, offsets), 0, 0);
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((value) => (value ? resolve(value) : reject(new Error('Canvas export failed.'))), 'image/png');
      });
      const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
      const tempPath = await invoke<string>(Invokes.SaveTempFile, { bytes });
      await invoke(Invokes.ExportImages, {
        baseOriginFolders: [],
        currentEditAdjustments: null,
        currentEditPath: tempPath,
        exportSettings: buildBackendExportSettings(settings, '{original_filename}_brightness'),
        isExplicitFilePath: true,
        outputFolderOrFile: outputPath,
        outputFormat: exportDialogExtension(settings.format),
        paths: [tempPath],
        waitForCompletion: true,
      });
      return outputPath;
    },
    [grid, offsets, osPlatform, source, t, autoEnabled, illumination, autoStrength, protectedAreas, manualGrid],
  );

  const selected = grid?.regions[selectedRegion] ?? null;
  const selectedOffset = offsets[selectedRegion] ?? 0;
  const maxBrightness = grid?.maximum ?? 255;
  const minBrightness = grid?.minimum ?? 0;
  const adjustedAverage = renderedGrid?.regions[selectedRegion]?.average ?? 0;

  return (
    <div className="brightness-tool-page">
      <header className="brightness-tool-header">
        <div className="brightness-tool-header-leading">
          <button
            aria-label={t('brightnessTool.back', { defaultValue: '返回图库' })}
            className="brightness-tool-back"
            onClick={onBack}
            type="button"
          >
            <ArrowLeft aria-hidden="true" size={18} />
          </button>
          <div>
            <span className="brightness-tool-eyebrow">
              {t('brightnessTool.eyebrow', { defaultValue: 'LIGHT MAP' })}
            </span>
            <h1>{t('brightnessTool.title', { defaultValue: '区域亮度工具' })}</h1>
            <p>{t('brightnessTool.description', { defaultValue: '查看整张图片的明暗分布，并分别校正每个区域。' })}</p>
          </div>
        </div>
        <div className="brightness-tool-header-actions">
          {source && (
            <Button onClick={() => fileInputRef.current?.click()} size="sm" variant="secondary">
              <ImagePlus aria-hidden="true" size={16} />
              {t('brightnessTool.changeImage', { defaultValue: '更换图片' })}
            </Button>
          )}
          <Button onClick={onBack} size="sm" variant="ghost">
            {t('brightnessTool.close', { defaultValue: '关闭' })}
          </Button>
        </div>
      </header>

      <input
        ref={fileInputRef}
        accept="image/*"
        className="brightness-tool-file-input"
        onChange={handleFileChange}
        type="file"
      />

      {!source ? (
        <div
          className={`brightness-tool-empty ${isDragging ? 'is-dragging' : ''}`}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setIsDragging(false);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
        >
          <div className="brightness-tool-empty-icon">
            <Sun aria-hidden="true" size={28} />
          </div>
          <h2>{t('brightnessTool.emptyTitle', { defaultValue: '导入一张图片开始分析' })}</h2>
          <p>
            {t('brightnessTool.emptyDescription', {
              defaultValue: '拖入图片，或选择文件。所有分析和调整都在本地完成。',
            })}
          </p>
          <Button onClick={() => fileInputRef.current?.click()} size="lg">
            <Upload aria-hidden="true" size={18} />
            {t('brightnessTool.chooseImage', { defaultValue: '选择图片' })}
          </Button>
          {error && (
            <p className="brightness-tool-error" role="alert">
              {error}
            </p>
          )}
        </div>
      ) : (
        <main className="brightness-tool-body">
          <canvas aria-hidden="true" className="brightness-tool-source-canvas" ref={sourceCanvasRef} />
          <section className="brightness-tool-preview-panel">
            <div className="brightness-tool-panel-heading">
              <div>
                <span className="brightness-tool-panel-kicker">
                  {t('brightnessTool.mapLabel', { defaultValue: 'BRIGHTNESS MAP' })}
                </span>
                <h2 title={source.name}>{source.name}</h2>
              </div>
              {grid && (
                <span className="brightness-tool-size">
                  {grid.columns} × {grid.rows} {t('brightnessTool.manualGrid', { defaultValue: '手动网格' })}
                </span>
              )}
            </div>
            <div
              aria-label={t('brightnessTool.preview', { defaultValue: '亮度调整预览' })}
              className="brightness-tool-preview-stage"
              onDoubleClick={() => (previewZoom > MIN_PREVIEW_ZOOM ? resetPreviewZoom() : applyPreviewZoom(2))}
              onKeyDown={handlePreviewKeyDown}
              onPointerCancel={handlePreviewPointerUp}
              onPointerDown={handlePreviewPointerDown}
              onPointerMove={handlePreviewPointerMove}
              onPointerUp={handlePreviewPointerUp}
              onWheel={handlePreviewWheel}
              ref={previewStageRef}
              role="region"
              tabIndex={0}
            >
              <div
                className="brightness-tool-zoom-content"
                style={{
                  height: imageBounds.height,
                  left: imageBounds.left,
                  top: imageBounds.top,
                  transform: `translate3d(${previewPan.x}px, ${previewPan.y}px, 0) scale(${previewZoom})`,
                  width: imageBounds.width,
                }}
              >
                <canvas aria-hidden="true" className="brightness-tool-preview-canvas" ref={previewCanvasRef} />
                {seamsVisible && (
                  <svg
                    className="brightness-tool-seam-overlay"
                    viewBox="0 0 1000 1000"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                  >
                    {seams
                      .filter((seam) => seam.id === selectedSeam && !ignoredSeams.includes(seam.id))
                      .map((seam) => (
                        <g
                          key={seam.id}
                          fill={
                            selectedSeam === seam.id
                              ? '#ffdf00'
                              : seam.nativeProfile?.status === 'supported'
                                ? '#22d3ee'
                                : '#94a3b8'
                          }
                        >
                          {seam.nativePath && (
                            <polyline
                              points={seam.nativePath.map(([x, y]) => `${x * 1000},${y * 1000}`).join(' ')}
                              fill="none"
                              stroke={selectedSeam === seam.id ? '#ffdf00' : '#94a3b8'}
                              strokeWidth={1}
                            />
                          )}
                          {!seam.nativePath &&
                            seam.points.map(([x, y], index) => (
                              <circle key={index} cx={x * 1000} cy={y * 1000} r={2} />
                            ))}
                          {seam.nativeLocations?.map(([x, y], index) => (
                            <circle
                              key={`native-${index}`}
                              cx={x * 1000}
                              cy={y * 1000}
                              r={3}
                              fill="none"
                              stroke="#fff"
                              strokeWidth={1}
                            />
                          ))}
                        </g>
                      ))}
                  </svg>
                )}
                {grid && isBrightnessLayerVisible && (
                  <div
                    className="brightness-tool-grid"
                    style={{
                      gridTemplateColumns: `repeat(${grid.columns}, 1fr)`,
                      gridTemplateRows: `repeat(${grid.rows}, 1fr)`,
                    }}
                  >
                    {grid.regions.map((region) => {
                      const isSelected = selectedRegion === region.index;
                      const offset = offsets[region.index] ?? 0;
                      return (
                        <button
                          aria-label={`${t('brightnessTool.region', { defaultValue: '区域' })} ${region.row + 1}-${region.column + 1}`}
                          aria-pressed={isSelected}
                          className={`brightness-tool-cell ${isSelected ? 'is-selected' : ''}`}
                          key={region.index}
                          onClick={() => setSelectedRegion(region.index)}
                          style={{
                            backgroundColor: heatColor(
                              renderedGrid?.regions[region.index]?.average ?? region.average,
                              minBrightness,
                              maxBrightness,
                            ),
                          }}
                          type="button"
                        >
                          <span>
                            {formatBrightness(renderedGrid?.regions[region.index]?.average ?? region.average)}
                          </span>
                          {offset !== 0 && (
                            <small>
                              {offset > 0 ? '+' : ''}
                              {Math.round(offset)}
                            </small>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
              <div
                aria-label={t('modals.batchGeometry.previewStageLabel', { defaultValue: '预览缩放控制' })}
                className="brightness-tool-zoom-toolbar"
                onDoubleClick={(event) => event.stopPropagation()}
                role="toolbar"
              >
                <button
                  aria-label={t('modals.batchGeometry.zoomOut', { defaultValue: '缩小预览' })}
                  className="brightness-tool-zoom-button"
                  disabled={previewZoom <= MIN_PREVIEW_ZOOM}
                  onClick={() => applyPreviewZoom(previewZoom / 1.25)}
                  type="button"
                >
                  <ZoomOut aria-hidden="true" size={16} />
                </button>
                <span className="brightness-tool-zoom-label">{Math.round(previewZoom * 100)}%</span>
                <button
                  aria-label={t('modals.batchGeometry.zoomIn', { defaultValue: '放大预览' })}
                  className="brightness-tool-zoom-button"
                  disabled={previewZoom >= MAX_PREVIEW_ZOOM}
                  onClick={() => applyPreviewZoom(previewZoom * 1.25)}
                  type="button"
                >
                  <ZoomIn aria-hidden="true" size={16} />
                </button>
                <button
                  aria-label={t('modals.batchGeometry.resetZoom', { defaultValue: '适应窗口' })}
                  className="brightness-tool-zoom-button"
                  onClick={resetPreviewZoom}
                  type="button"
                >
                  <ScanSearch aria-hidden="true" size={16} />
                </button>
              </div>
            </div>
            {isBrightnessLayerVisible && (
              <div className="brightness-tool-legend">
                <span>{t('brightnessTool.dark', { defaultValue: '暗' })}</span>
                <div className="brightness-tool-gradient" />
                <span>{t('brightnessTool.bright', { defaultValue: '亮' })}</span>
                <span className="brightness-tool-legend-note">
                  {t('brightnessTool.legendNote', { defaultValue: '数字为区域平均亮度（0–255）' })}
                </span>
              </div>
            )}
          </section>

          <div className="brightness-tool-seam-controls">
            <Button size="sm" variant="secondary" onClick={analyzeSeams} disabled={seamsLoading}>
              {seamsLoading ? seamProgress : t('brightnessTool.detectSeams', { defaultValue: '接缝诊断（实验）' })}
            </Button>
            {seamsAnalyzed && (
              <>
                <label>
                  <input
                    type="checkbox"
                    checked={seamsVisible}
                    onChange={(event) => setSeamsVisible(event.target.checked)}
                  />
                  {t('brightnessTool.seamOverlay', { defaultValue: '候选标记' })}
                </label>
                <select
                  aria-label={t('brightnessTool.seamCandidate', { defaultValue: '候选接缝' })}
                  value={selectedSeam ?? ''}
                  onChange={(event) => setSelectedSeam(event.target.value === '' ? null : Number(event.target.value))}
                >
                  {!seams.length && (
                    <option value="">{t('brightnessTool.noSeams', { defaultValue: '未发现可靠候选' })}</option>
                  )}
                  {seams.map((seam) => (
                    <option key={seam.id} value={seam.id}>
                      {t('brightnessTool.nativeCandidateLabel', {
                        defaultValue: '#{{id}} · {{evidence}} · {{supported}}/{{total}}',
                        id: seam.id + 1,
                        supported: seam.nativeProfile?.supported ?? 0,
                        total: seam.nativeProfile?.samples ?? 0,
                        evidence:
                          seam.nativeProfile?.status === 'supported'
                            ? t('brightnessTool.nativeProfile_supported', { defaultValue: '局部阶跃与纹理相容' })
                            : t('brightnessTool.nativeProfile_insufficient', { defaultValue: '接缝证据不足' }),
                      })}
                    </option>
                  ))}
                </select>
                {selectedSeam !== null && (
                  <label>
                    <input
                      type="checkbox"
                      checked={ignoredSeams.includes(selectedSeam)}
                      onChange={(event) =>
                        setIgnoredSeams((current) =>
                          event.target.checked
                            ? [...current, selectedSeam]
                            : current.filter((id) => id !== selectedSeam),
                        )
                      }
                    />
                    {t('brightnessTool.ignoreSeam', { defaultValue: '排除此候选' })}
                  </label>
                )}
                <span>
                  {t('brightnessTool.nativeDiagnosticOnly', { defaultValue: '空心点：源尺寸复核位置；仅诊断' })}
                </span>
              </>
            )}
          </div>
          <div className="brightness-tool-auto-controls">
            <label>
              <input type="checkbox" checked={autoEnabled} onChange={(event) => setAutoEnabled(event.target.checked)} />
              {t('brightnessTool.adaptive', { defaultValue: '自适应校正' })}
            </label>
            <label htmlFor="brightness-auto-strength">{t('brightnessTool.strength', { defaultValue: '强度' })}</label>
            <input
              id="brightness-auto-strength"
              type="range"
              min="0"
              max="100"
              value={autoStrength}
              onChange={(event) => setAutoStrength(Number(event.target.value))}
              disabled={!autoEnabled}
            />
            <output>{autoStrength}%</output>
            <label>
              <input
                type="checkbox"
                checked={excludedRegions.includes(selectedRegion)}
                onChange={(event) =>
                  setExcludedRegions((current) =>
                    event.target.checked
                      ? [...current, selectedRegion]
                      : current.filter((index) => index !== selectedRegion),
                  )
                }
              />
              {t('brightnessTool.protectRegion', { defaultValue: '当前区域不参与自动调整' })}
            </label>
          </div>
          <aside className="brightness-tool-controls">
            <div className="brightness-tool-control-card">
              <div className="brightness-tool-control-heading">
                <div>
                  <span className="brightness-tool-panel-kicker">
                    {t('brightnessTool.selectedRegion', { defaultValue: 'SELECTED REGION' })}
                  </span>
                  <h3>
                    {selected
                      ? `${t('brightnessTool.region', { defaultValue: '区域' })} ${selected.row + 1}-${selected.column + 1}`
                      : '—'}
                  </h3>
                </div>
                <strong>{formatBrightness(adjustedAverage)}</strong>
              </div>
              <label className="brightness-tool-range-label" htmlFor="brightness-region-range">
                <span>{t('brightnessTool.adjustment', { defaultValue: '亮度调整' })}</span>
                <output>
                  {selectedOffset > 0 ? '+' : ''}
                  {Math.round(selectedOffset)}
                </output>
              </label>
              <input
                aria-label={t('brightnessTool.adjustment', { defaultValue: '亮度调整' })}
                className="brightness-tool-range"
                id="brightness-region-range"
                max="100"
                min="-100"
                onChange={(event) => updateOffset(selectedRegion, Number(event.target.value))}
                step="1"
                type="range"
                value={selectedOffset}
              />
              <div className="brightness-tool-range-labels">
                <span>{t('brightnessTool.rangeMin', { defaultValue: '−100' })}</span>
                <span>{t('brightnessTool.rangeZero', { defaultValue: '0' })}</span>
                <span>{t('brightnessTool.rangeMax', { defaultValue: '+100' })}</span>
              </div>
              {selected && (
                <dl className="brightness-tool-stats">
                  <div>
                    <dt>{t('brightnessTool.original', { defaultValue: '原始平均' })}</dt>
                    <dd>{formatBrightness(selected.average)}</dd>
                  </div>
                  <div>
                    <dt>{t('brightnessTool.range', { defaultValue: '区域范围' })}</dt>
                    <dd>
                      {formatBrightness(selected.minimum)}–{formatBrightness(selected.maximum)}
                    </dd>
                  </div>
                </dl>
              )}
            </div>
            <div className="brightness-tool-action-stack">
              <div className="brightness-tool-control-actions">
                <Button className="brightness-tool-uniform" onClick={normalizeImage} size="sm">
                  <Grid2X2 aria-hidden="true" size={16} />
                  {t('brightnessTool.uniform', { defaultValue: '统一亮度' })}
                </Button>
                <Button
                  aria-label={t('brightnessTool.reset', { defaultValue: '重置调整' })}
                  className="brightness-tool-reset"
                  onClick={resetOffsets}
                  size="icon"
                  variant="secondary"
                >
                  <RefreshCcw aria-hidden="true" size={16} />
                </Button>
              </div>
              <Button
                aria-pressed={isBrightnessLayerVisible}
                className="brightness-tool-layer-toggle"
                onClick={() => setIsBrightnessLayerVisible((visible) => !visible)}
                size="sm"
                variant={isBrightnessLayerVisible ? 'secondary' : 'primary'}
              >
                {isBrightnessLayerVisible ? (
                  <EyeOff aria-hidden="true" size={16} />
                ) : (
                  <Eye aria-hidden="true" size={16} />
                )}
                {t(isBrightnessLayerVisible ? 'brightnessTool.layerOff' : 'brightnessTool.layerOn', {
                  defaultValue: isBrightnessLayerVisible ? '隐藏图层' : '显示图层',
                })}
              </Button>
              {error && (
                <p className="brightness-tool-error brightness-tool-action-error" role="alert">
                  {error}
                </p>
              )}
              <Button
                className="brightness-tool-download"
                disabled={!renderedImageData}
                onClick={() => setIsExportDialogOpen(true)}
                size="sm"
              >
                {t('brightnessTool.download', { defaultValue: '导出图片' })}
              </Button>
            </div>
          </aside>
        </main>
      )}
      <ExportImageDialog
        initialFormat="jpeg"
        isOpen={isExportDialogOpen}
        metadata={source?.metadata ?? null}
        onClose={() => setIsExportDialogOpen(false)}
        onExport={handleExport}
        source={exportSource}
      />
    </div>
  );
}
