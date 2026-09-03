export const clampComparePosition = (position: number) => Math.min(1, Math.max(0, position));

export const comparePositionFromClientX = (clientX: number, left: number, width: number): number | null => {
  if (!Number.isFinite(clientX) || !Number.isFinite(left) || !Number.isFinite(width) || width <= 0) return null;
  return clampComparePosition((clientX - left) / width);
};

export const comparePositionFromKey = (position: number, key: string, accelerated = false): number | null => {
  if (key === 'Home') return 0;
  if (key === 'End') return 1;

  const step = accelerated ? 0.1 : 0.01;
  const delta =
    key === 'ArrowLeft' || key === 'ArrowDown'
      ? -step
      : key === 'ArrowRight' || key === 'ArrowUp'
        ? step
        : key === 'PageDown'
          ? -0.1
          : key === 'PageUp'
            ? 0.1
            : null;

  return delta === null ? null : clampComparePosition(position + delta);
};
