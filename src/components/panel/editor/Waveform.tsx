import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence, LayoutGroup } from 'framer-motion';
import { AlertOctagon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { WaveformData } from '../../ui/AppProperties';
import { DisplayMode } from '../../../utils/adjustments';

interface WaveformProps {
  waveformData: WaveformData | null;
  histogram?: any;
  displayMode: string;
  setDisplayMode: (mode: string) => void;
  histogramOnly?: boolean;
  showClipping?: boolean;
  onToggleClipping?: () => void;
  theme?: string;
}

const modeButtons = [
  {
    mode: DisplayMode.Luma,
    label: 'L',
    tooltip: 'ui.waveform.tooltips.luma',
    bgClass: 'bg-accent',
    textActiveClass: 'text-button-text',
  },
  {
    mode: DisplayMode.Rgb,
    label: 'RGB',
    tooltip: 'ui.waveform.tooltips.rgb',
    bgClass: 'bg-accent',
    textActiveClass: 'text-button-text',
  },
  {
    mode: DisplayMode.Parade,
    label: 'P',
    tooltip: 'ui.waveform.tooltips.parade',
    bgClass: 'bg-accent',
    textActiveClass: 'text-button-text',
  },
  {
    mode: DisplayMode.Vectorscope,
    label: 'V',
    tooltip: 'ui.waveform.tooltips.vectorscope',
    bgClass: 'bg-accent',
    textActiveClass: 'text-button-text',
  },
  {
    mode: DisplayMode.Histogram,
    label: 'H',
    tooltip: 'ui.waveform.tooltips.histogram',
    bgClass: 'bg-accent',
    textActiveClass: 'text-button-text',
  },
];

const HISTOGRAM_WIDTH = 256;
const HISTOGRAM_HEIGHT = 96;
const HISTOGRAM_TOP = 5;
const HISTOGRAM_BASELINE = 92;

const HistogramView = ({ histogram, label }: { histogram: any; label: string }) => {
  if (!histogram || !histogram.red || !histogram.green || !histogram.blue) return null;

  const getPath = (data: number[], closePath: boolean) => {
    const values = Array.isArray(data) ? data : [];
    if (values.length === 0) return '';

    const points = values
      .map((value, index) => {
        const normalizedX = values.length === 1 ? 0 : index / (values.length - 1);
        const normalizedValue = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
        const x = normalizedX * HISTOGRAM_WIDTH;
        const y = HISTOGRAM_BASELINE - normalizedValue * (HISTOGRAM_BASELINE - HISTOGRAM_TOP);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(' L');

    return closePath ? `M0,${HISTOGRAM_BASELINE} L${points} L${HISTOGRAM_WIDTH},${HISTOGRAM_BASELINE} Z` : `M${points}`;
  };

  const channels = [
    { key: 'red', color: '#ff4f58', data: histogram.red },
    { key: 'green', color: '#42d474', data: histogram.green },
    { key: 'blue', color: '#4f8cff', data: histogram.blue },
  ];

  return (
    <svg
      aria-label={label}
      className="histogram-chart pointer-events-none"
      preserveAspectRatio="none"
      role="img"
      viewBox={`0 0 ${HISTOGRAM_WIDTH} ${HISTOGRAM_HEIGHT}`}
    >
      <g aria-hidden="true" className="histogram-grid">
        {[64, 128, 192].map((x) => (
          <line key={`vertical-${x}`} x1={x} x2={x} y1={HISTOGRAM_TOP} y2={HISTOGRAM_BASELINE} />
        ))}
        <line x1="0" x2={HISTOGRAM_WIDTH} y1="48" y2="48" />
        <line
          className="histogram-baseline"
          x1="0"
          x2={HISTOGRAM_WIDTH}
          y1={HISTOGRAM_BASELINE}
          y2={HISTOGRAM_BASELINE}
        />
      </g>
      {channels.map((ch) => {
        if (!ch.data || ch.data.length === 0) return null;
        return (
          <g key={ch.key} className="histogram-channel">
            <path d={getPath(ch.data, true)} fill={ch.color} fillOpacity={0.52} />
            <path
              d={getPath(ch.data, false)}
              fill="none"
              stroke={ch.color}
              strokeOpacity={0.92}
              strokeWidth={1.1}
              strokeLinejoin="round"
              vectorEffect="non-scaling-stroke"
            />
          </g>
        );
      })}
    </svg>
  );
};

const HistogramLoader = () => (
  <div aria-hidden="true" className="histogram-loader">
    <span className="histogram-loader-sweep" />
  </div>
);

const useRawRgbaCanvas = (
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  base64Data: string,
  width: number,
  height: number,
) => {
  useEffect(() => {
    if (!base64Data || !canvasRef.current || !width || !height) return;

    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.imageSmoothingEnabled = false;

    const binary = atob(base64Data);
    const bytes = new Uint8ClampedArray(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }

    const imageData = new ImageData(bytes, width, height);
    ctx.putImageData(imageData, 0, 0);
  }, [base64Data, width, height, canvasRef]);
};

const WaveformCanvas = ({
  base64Data,
  width,
  height,
  isVectorscope,
}: {
  base64Data: string;
  width: number;
  height: number;
  isVectorscope: boolean;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useRawRgbaCanvas(canvasRef, base64Data, width, height);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={`w-full h-full ${isVectorscope ? 'object-contain' : ''}`}
    />
  );
};

const FakeWaveformLoader = ({ mode }: { mode: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastTimeRef = useRef<number>(0);
  const spawnAccumulatorRef = useRef<number>(0);

  const MAX_PARTICLES = 10000;
  const particles = useRef(
    Array.from({ length: MAX_PARTICLES }, () => ({
      x: 0,
      y: 0,
      targetX: 0,
      targetY: 0,
      life: 0,
      maxLife: 1,
      r: 255,
      g: 255,
      b: 255,
      active: false,
    })),
  ).current;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    const WIDTH = 256;
    const HEIGHT = 256;
    canvas.width = WIDTH;
    canvas.height = HEIGHT;

    particles.forEach((p) => (p.active = false));
    lastTimeRef.current = 0;
    spawnAccumulatorRef.current = 0;
    let isPrewarmed = false;

    const imgData = ctx.createImageData(WIDTH, HEIGHT);
    const data = imgData.data;

    const gridBuffer = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
    if (mode === DisplayMode.Vectorscope) {
      for (let py = 0; py < HEIGHT; py++) {
        for (let px = 0; px < WIDTH; px++) {
          const dx = px - 128.0;
          const dy = 128.0 - py;
          const min_d = Math.min(Math.abs(dx), Math.abs(dy));
          const dist = Math.sqrt(dx * dx + dy * dy);
          const off = (py * WIDTH + px) * 4;

          if (min_d <= 1.0) {
            const alpha = Math.max(0, 40.0 - min_d * 30.0);
            gridBuffer[off] = 255;
            gridBuffer[off + 1] = 255;
            gridBuffer[off + 2] = 255;
            gridBuffer[off + 3] = alpha;
          } else if (Math.abs(dist - 127.0) < 0.8 || Math.abs(dist - 64.0) < 0.8) {
            gridBuffer[off] = 255;
            gridBuffer[off + 1] = 255;
            gridBuffer[off + 2] = 255;
            gridBuffer[off + 3] = 15;
          } else if (dx < 0.0 && dy > 0.0 && Math.abs(dy + 1.53 * dx) < 1.0) {
            gridBuffer[off] = 255;
            gridBuffer[off + 1] = 200;
            gridBuffer[off + 2] = 150;
            gridBuffer[off + 3] = 120;
          }
        }
      }
    }

    let animationFrameId: number;

    const render = (time: number) => {
      if (lastTimeRef.current === 0) lastTimeRef.current = time;
      let dt = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;

      if (dt > 0.05) dt = 0.05;

      let frameDt = dt;

      if (!isPrewarmed) {
        frameDt = 0.5;
        isPrewarmed = true;
      }

      data.set(gridBuffer);

      const SPAWN_RATE = 1000;
      spawnAccumulatorRef.current += SPAWN_RATE * frameDt;

      const dotsToSpawn = Math.floor(spawnAccumulatorRef.current);
      if (dotsToSpawn > 0) {
        spawnAccumulatorRef.current -= dotsToSpawn;
        let spawnedCount = 0;

        for (let i = 0; i < MAX_PARTICLES && spawnedCount < dotsToSpawn; i++) {
          const p = particles[i];
          if (!p.active) {
            p.active = true;

            if (mode !== DisplayMode.Vectorscope) {
              p.x = Math.random() * WIDTH;
              p.targetX = p.x;

              p.y = HEIGHT - Math.random() * 2;

              const isPot = Math.random() < 0.4;

              if (isPot) {
                p.targetY = HEIGHT - Math.random() * 6;
              } else {
                const randomCurve = Math.pow(Math.random(), 1.2);
                p.targetY = HEIGHT - randomCurve * HEIGHT;
              }

              if (mode === DisplayMode.Parade) {
                const section = Math.floor((p.x / WIDTH) * 3);
                if (section === 0) {
                  p.r = 255;
                  p.g = 70;
                  p.b = 70;
                } else if (section === 1) {
                  p.r = 70;
                  p.g = 255;
                  p.b = 70;
                } else {
                  p.r = 70;
                  p.g = 150;
                  p.b = 255;
                }
              } else if (mode === DisplayMode.Rgb) {
                const rand = Math.random();
                if (rand > 0.85) {
                  p.r = 255;
                  p.g = 80;
                  p.b = 80;
                } else if (rand > 0.7) {
                  p.r = 80;
                  p.g = 255;
                  p.b = 80;
                } else if (rand > 0.55) {
                  p.r = 80;
                  p.g = 150;
                  p.b = 255;
                } else {
                  p.r = 255;
                  p.g = 255;
                  p.b = 255;
                }
              } else {
                p.r = 255;
                p.g = 255;
                p.b = 255;
              }
            }

            const life = Math.random() * 5.0 + 8.0;
            p.life = life;
            p.maxLife = life;

            spawnedCount++;
          }
        }
      }

      const speedMultiplier = 1;
      const interpolation = 1 - Math.exp(-speedMultiplier * frameDt);

      for (let i = 0; i < MAX_PARTICLES; i++) {
        const p = particles[i];
        if (p.active) {
          p.life -= frameDt;

          if (p.life <= 0) {
            p.active = false;
            continue;
          }

          p.x += (p.targetX - p.x) * interpolation;
          p.y += (p.targetY - p.y) * interpolation;

          const lifeRatio = p.life / p.maxLife;

          let opacity = 1.0;
          if (lifeRatio > 0.8) {
            opacity = (1 - lifeRatio) / 0.12;
          } else if (lifeRatio < 0.2) {
            opacity = lifeRatio / 0.2;
          }

          const alpha = opacity * 0.18;

          const px = Math.floor(p.x);
          const py = Math.floor(p.y);

          for (let dy = 0; dy < 3; dy++) {
            for (let dx = 0; dx < 1.5; dx++) {
              const cx = px + dx;
              const cy = py + dy;

              if (cx >= 0 && cx < WIDTH && cy >= 0 && cy < HEIGHT) {
                const idx = (cy * WIDTH + cx) * 4;
                data[idx] = Math.min(255, data[idx] + p.r * alpha);
                data[idx + 1] = Math.min(255, data[idx + 1] + p.g * alpha);
                data[idx + 2] = Math.min(255, data[idx + 2] + p.b * alpha);
                data[idx + 3] = Math.min(255, data[idx + 3] + alpha * 255);
              }
            }
          }
        }
      }

      ctx.putImageData(imgData, 0, 0);
      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => cancelAnimationFrame(animationFrameId);
  }, [mode, particles]);

  return (
    <canvas ref={canvasRef} className={`w-full h-full ${mode === DisplayMode.Vectorscope ? 'object-contain' : ''}`} />
  );
};

export default function Waveform({
  waveformData,
  histogram,
  displayMode,
  setDisplayMode,
  histogramOnly = false,
  showClipping,
  onToggleClipping,
  theme,
}: WaveformProps) {
  const { t } = useTranslation();
  const [isHovered, setIsHovered] = useState(false);
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const effectiveDisplayMode = histogramOnly ? DisplayMode.Histogram : displayMode;
  const isLightTheme = theme ? ['light', 'snow', 'arctic'].includes(theme) : false;
  const isHistogram = effectiveDisplayMode === DisplayMode.Histogram;
  const isVectorscope = effectiveDisplayMode === DisplayMode.Vectorscope;
  const isReady = isHistogram ? !!(histogram && histogram.red) : !!waveformData;
  const hadDataOnMount = useRef(isReady);
  const width = waveformData?.width || 256;
  const height = waveformData?.height || 256;

  const activeData = waveformData
    ? {
        [DisplayMode.Rgb]: waveformData.rgb,
        [DisplayMode.Luma]: waveformData.luma,
        [DisplayMode.Parade]: waveformData.parade,
        [DisplayMode.Vectorscope]: waveformData.vectorscope,
        [DisplayMode.Histogram]: undefined,
      }[effectiveDisplayMode as DisplayMode]
    : '';

  const handleMouseEnter = () => {
    hoverTimeoutRef.current = setTimeout(() => {
      setIsHovered(true);
    }, 250);
  };

  const handleMouseLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setIsHovered(false);
  };

  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) {
        clearTimeout(hoverTimeoutRef.current);
      }
    };
  }, []);

  const baseButtonClass =
    'relative grow text-center px-1 py-0.5 text-[10px] rounded-[2px] font-medium transition-colors duration-100';
  const inactiveButtonClass = 'text-text-primary hover:bg-bg-tertiary';

  const isLoaderMode = [
    DisplayMode.Luma,
    DisplayMode.Rgb,
    DisplayMode.Parade,
    DisplayMode.Vectorscope,
    DisplayMode.Histogram,
  ].includes(effectiveDisplayMode as DisplayMode);

  const clippingLabel = showClipping ? t('ui.waveform.tooltips.hideClipping') : t('ui.waveform.tooltips.showClipping');
  const histogramLabel = t('ui.waveform.tooltips.histogram');

  return (
    <div
      className={`relative w-full h-full overflow-hidden border ${
        isHistogram ? 'histogram-surface' : 'bg-surface rounded-[2px] border-border-color'
      }`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div
        className="absolute inset-0 z-0 pointer-events-none"
        style={{
          isolation: 'isolate',
          filter: !isHistogram && isLightTheme ? 'invert(1) hue-rotate(180deg)' : 'none',
          transition: 'filter 0.3s ease',
        }}
      >
        <AnimatePresence initial={!hadDataOnMount.current} mode="sync">
          {isReady ? (
            isHistogram ? (
              <motion.div
                key="waveform-histogram"
                initial={{ opacity: 0, scaleY: 0 }}
                animate={{ opacity: 1, scaleY: 1 }}
                exit={{ opacity: 0, scaleY: 0 }}
                transition={{
                  duration: 0.18,
                  ease: [0.22, 1, 0.36, 1],
                  opacity: { duration: 0.15 },
                }}
                style={{ transformOrigin: 'bottom' }}
                className="waveform-histogram-layer absolute z-10"
              >
                <HistogramView histogram={histogram} label={histogramLabel} />
              </motion.div>
            ) : (
              <motion.div
                key={`waveform-canvas-${effectiveDisplayMode}`}
                initial={{ opacity: 0, ...(isVectorscope ? {} : { scaleY: 0 }) }}
                animate={{ opacity: 1, ...(isVectorscope ? {} : { scaleY: 1 }) }}
                exit={{ opacity: 0, ...(isVectorscope ? {} : { scaleY: 0 }) }}
                transition={{
                  duration: 0.18,
                  ease: [0.22, 1, 0.36, 1],
                  opacity: { duration: 0.15 },
                }}
                style={{ transformOrigin: 'bottom' }}
                className="absolute inset-0 z-10"
              >
                <WaveformCanvas
                  base64Data={activeData || ''}
                  width={width}
                  height={height}
                  isVectorscope={isVectorscope}
                />
              </motion.div>
            )
          ) : isLoaderMode ? (
            <motion.div
              key={`waveform-loader-${effectiveDisplayMode}`}
              initial={{ opacity: 0 }}
              animate={{
                opacity: 1,
                transition: { duration: 0.18, ease: 'easeOut' },
              }}
              exit={{
                opacity: 0,
                transition: { duration: 0.2, ease: 'easeIn' },
              }}
              className="absolute inset-0 pointer-events-none z-0"
            >
              {isHistogram ? <HistogramLoader /> : <FakeWaveformLoader mode={effectiveDisplayMode} />}
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>

      {isHistogram && (
        <div aria-hidden="true" className="histogram-tonal-axis">
          <span>{t('adjustments.basic.shadows')}</span>
          <span>{t('adjustments.color.grading.midtones')}</span>
          <span>{t('adjustments.basic.highlights')}</span>
        </div>
      )}

      {isHistogram && onToggleClipping && (
        <>
          <button
            aria-label={`${t('adjustments.basic.shadows')} · ${clippingLabel}`}
            aria-pressed={showClipping}
            className={`histogram-clipping-toggle is-shadow ${showClipping ? 'is-active' : ''}`}
            data-tooltip={`${t('adjustments.basic.shadows')} · ${clippingLabel}`}
            onClick={onToggleClipping}
            type="button"
          >
            <svg aria-hidden="true" viewBox="0 0 16 14">
              <path d="M1.5 12.5 8 1.5l6.5 11Z" />
            </svg>
          </button>
          <button
            aria-label={`${t('adjustments.basic.highlights')} · ${clippingLabel}`}
            aria-pressed={showClipping}
            className={`histogram-clipping-toggle is-highlight ${showClipping ? 'is-active' : ''}`}
            data-tooltip={`${t('adjustments.basic.highlights')} · ${clippingLabel}`}
            onClick={onToggleClipping}
            type="button"
          >
            <svg aria-hidden="true" viewBox="0 0 16 14">
              <path d="M1.5 12.5 8 1.5l6.5 11Z" />
            </svg>
          </button>
        </>
      )}

      <AnimatePresence>
        {isHovered && !histogramOnly && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
            className="absolute inset-x-0 bottom-0 p-1 pt-5 bg-linear-to-t from-black/80 to-transparent flex justify-center z-20"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.1, ease: 'easeOut', delay: 0.05 }}
              className="flex items-center justify-center gap-0.5 p-0.5 bg-surface/96 rounded-[3px] w-full border border-border-color"
            >
              {onToggleClipping && !isHistogram && (
                <>
                  <button
                    onClick={onToggleClipping}
                    data-tooltip={clippingLabel}
                    className={`relative flex items-center justify-center w-6 h-6 shrink-0 rounded-[2px] transition-colors duration-100 ${
                      showClipping ? 'bg-accent text-button-text' : 'text-text-primary hover:bg-bg-tertiary'
                    }`}
                  >
                    <AlertOctagon size={14} />
                  </button>
                  <div className="w-px h-5 bg-white/20 mx-1 shrink-0"></div>
                </>
              )}

              <LayoutGroup>
                {modeButtons.map(({ mode, label, tooltip, bgClass, textActiveClass }) => (
                  <button
                    key={mode}
                    onClick={() => setDisplayMode(mode)}
                    data-tooltip={t(tooltip as never) as string}
                    className={`${baseButtonClass} ${effectiveDisplayMode === mode ? textActiveClass : inactiveButtonClass}`}
                  >
                    {effectiveDisplayMode === mode && (
                      <motion.div
                        layoutId="waveform-mode-indicator"
                        className={`absolute inset-0 ${bgClass} rounded-[2px]`}
                        transition={{ duration: 0.12, ease: [0.22, 1, 0.36, 1] }}
                      />
                    )}
                    <span className="relative z-10">{label}</span>
                  </button>
                ))}
              </LayoutGroup>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
