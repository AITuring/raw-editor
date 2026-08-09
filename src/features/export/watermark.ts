import { convertFileSrc } from '@tauri-apps/api/core';

import defaultWatermarkUrl from '../../assets/default-watermark.png';

export const DEFAULT_WATERMARK_PATH = 'builtin://default-watermark';
export const DEFAULT_WATERMARK_URL = defaultWatermarkUrl;

export function isDefaultWatermarkPath(path: string | null | undefined): boolean {
  return !path || path === DEFAULT_WATERMARK_PATH;
}

export function normalizeWatermarkPath(path: string | null | undefined): string {
  return path && path !== DEFAULT_WATERMARK_PATH ? path : DEFAULT_WATERMARK_PATH;
}

export function getWatermarkPreviewUrl(path: string | null | undefined): string {
  const normalizedPath = normalizeWatermarkPath(path);
  return normalizedPath === DEFAULT_WATERMARK_PATH ? DEFAULT_WATERMARK_URL : convertFileSrc(normalizedPath);
}
