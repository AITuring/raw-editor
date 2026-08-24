export type MessageTaskKind = 'denoise' | 'hdr' | 'panorama';

export interface MessageTaskProgress {
  exact: boolean;
  value: number | null;
}

const clampPercentage = (value: number) => Math.min(100, Math.max(0, value));

export function getMessageTaskProgress(message: string | null | undefined, kind: MessageTaskKind): MessageTaskProgress {
  if (!message) return { exact: false, value: null };

  const percentageMatch = message.match(/(?:^|[^\d])(\d{1,3}(?:\.\d+)?)\s*%/);
  if (percentageMatch) {
    return { exact: true, value: clampPercentage(Number(percentageMatch[1])) };
  }

  const normalized = message.toLowerCase();

  if (kind === 'denoise') {
    if (normalized.includes('generating preview')) return { exact: false, value: 99 };
    if (normalized.includes('finalizing')) return { exact: false, value: 99 };
    if (normalized.includes('blending detail')) return { exact: false, value: 99 };
    return { exact: false, value: null };
  }

  if (kind === 'panorama') {
    const stitchMatch = normalized.match(/(?:stitching image|focus-stacking image)\s+(\d+)\s+of\s+(\d+)/);
    if (stitchMatch) {
      const current = Number(stitchMatch[1]);
      const total = Number(stitchMatch[2]);
      if (total > 0) return { exact: false, value: clampPercentage(55 + (current / total) * 34) };
    }
    if (normalized.includes('creating preview')) return { exact: false, value: 98 };
    if (normalized.includes('finalizing')) return { exact: false, value: 94 };
    if (normalized.includes('warping') || normalized.includes('blending')) return { exact: false, value: 55 };
    if (normalized.includes('stitching order')) return { exact: false, value: 48 };
    if (normalized.includes('determining')) return { exact: false, value: 44 };
    if (normalized.includes('finding image matches')) return { exact: false, value: 30 };
    if (normalized.includes('processing')) return { exact: false, value: 15 };
    if (normalized.includes('loading and preparing')) return { exact: false, value: 8 };
    if (normalized.includes('starting')) return { exact: false, value: 3 };
    return { exact: false, value: null };
  }

  if (normalized.includes('creating preview')) return { exact: false, value: 98 };
  if (normalized.includes('tone mapping')) return { exact: false, value: 86 };
  if (normalized.includes('merging exposures')) return { exact: false, value: 68 };
  if (normalized.includes('could not align')) return { exact: false, value: 42 };
  if (normalized.includes('aligning')) return { exact: false, value: 42 };
  if (normalized.includes('deghosting')) return { exact: false, value: 28 };
  if (normalized.includes('processing')) return { exact: false, value: 12 };
  if (normalized.includes('starting')) return { exact: false, value: 3 };
  return { exact: false, value: null };
}
