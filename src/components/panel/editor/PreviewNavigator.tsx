import clsx from 'clsx';
import { useMemo, useRef } from 'react';
import type { KeyboardEvent, PointerEvent, RefObject } from 'react';

interface PreviewNavigatorProps {
  aspectRatio: number;
  label: string;
  onNavigate(normalizedX: number, normalizedY: number): void;
  onNudge(deltaX: number, deltaY: number): void;
  previewUrl: string;
  viewportRef: RefObject<HTMLDivElement | null>;
  visible: boolean;
  zoomPercent: number;
}

const MAX_NAVIGATOR_WIDTH = 168;
const MAX_NAVIGATOR_HEIGHT = 116;

export default function PreviewNavigator({
  aspectRatio,
  label,
  onNavigate,
  onNudge,
  previewUrl,
  viewportRef,
  visible,
  zoomPercent,
}: PreviewNavigatorProps) {
  const isDraggingRef = useRef(false);
  const size = useMemo(() => {
    const ratio = Math.min(12, Math.max(1 / 12, Number.isFinite(aspectRatio) ? aspectRatio : 1));
    if (ratio >= MAX_NAVIGATOR_WIDTH / MAX_NAVIGATOR_HEIGHT) {
      return { height: MAX_NAVIGATOR_WIDTH / ratio, width: MAX_NAVIGATOR_WIDTH };
    }
    return { height: MAX_NAVIGATOR_HEIGHT, width: MAX_NAVIGATOR_HEIGHT * ratio };
  }, [aspectRatio]);

  const navigateFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    onNavigate(
      Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width)),
      Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height)),
    );
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    isDraggingRef.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    navigateFromPointer(event);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    navigateFromPointer(event);
  };

  const handlePointerEnd = (event: PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    isDraggingRef.current = false;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 0.12 : 0.04;
    const delta =
      event.key === 'ArrowLeft'
        ? [-step, 0]
        : event.key === 'ArrowRight'
          ? [step, 0]
          : event.key === 'ArrowUp'
            ? [0, -step]
            : event.key === 'ArrowDown'
              ? [0, step]
              : null;
    if (!delta) return;
    event.preventDefault();
    event.stopPropagation();
    onNudge(delta[0], delta[1]);
  };

  return (
    <div
      aria-label={`${label} · ${zoomPercent}%`}
      className={clsx(
        'absolute bottom-4 left-4 z-30 rounded-[4px] border border-border-color bg-bg-primary/95 p-[3px]',
        'shadow-[0_6px_20px_rgba(0,0,0,0.28)] transition-[opacity,transform] duration-150 ease-out',
        visible ? 'pointer-events-auto translate-y-0 opacity-100' : 'pointer-events-none translate-y-1 opacity-0',
      )}
      onKeyDown={handleKeyDown}
      role="group"
      tabIndex={visible ? 0 : -1}
      title={`${label} · ${zoomPercent}%`}
    >
      <div
        className="relative touch-none overflow-hidden rounded-[2px] bg-bg-secondary"
        onClick={(event) => event.stopPropagation()}
        onPointerCancel={handlePointerEnd}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerEnd}
        style={{ height: size.height, width: size.width }}
      >
        <img alt="" className="absolute inset-0 h-full w-full object-fill" draggable={false} src={previewUrl} />
        <div
          className="absolute border-[1.5px] border-accent bg-accent/10 shadow-[0_0_0_1px_rgba(0,0,0,0.42)]"
          ref={viewportRef}
          style={{ height: '100%', left: 0, top: 0, width: '100%' }}
        />
      </div>
    </div>
  );
}
