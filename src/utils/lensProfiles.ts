import { invoke } from '@tauri-apps/api/core';

export interface LensDistortionParams {
  k1: number;
  k2: number;
  k3: number;
  model: number;
  tca_vr: number;
  tca_vb: number;
  vig_k1: number;
  vig_k2: number;
  vig_k3: number;
}

export interface DetectedLensProfile {
  maker: string;
  model: string;
  params: LensDistortionParams | null;
}

type ExifData = Record<string, unknown> | null | undefined;

const MAX_DETECTION_CACHE_ENTRIES = 64;
const detectionCache = new Map<string, Promise<DetectedLensProfile | null>>();

const readExifText = (exif: ExifData, ...keys: string[]): string => {
  for (const key of keys) {
    const value = exif?.[key];
    if (typeof value === 'string' && value.trim()) return value.trim();
    if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  }
  return '';
};

export const parseExifNumber = (value: unknown): number | null => {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value !== 'string') return null;

  const match = value.replace(',', '.').match(/-?\d+(?:\.\d+)?/);
  if (!match) return null;

  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : null;
};

export const detectLensProfileFromExif = (exif: ExifData): Promise<DetectedLensProfile | null> => {
  const maker = readExifText(exif, 'LensMake', 'Make');
  const model = readExifText(exif, 'LensModel', 'LensID', 'Lens');

  if (!model) return Promise.resolve(null);

  const focalLength = parseExifNumber(exif?.FocalLength) ?? parseExifNumber(exif?.FocalLengthIn35mmFilm) ?? 50;
  const aperture = parseExifNumber(exif?.FNumber) ?? parseExifNumber(exif?.ApertureValue);
  const distance = parseExifNumber(exif?.SubjectDistance);
  const cacheKey = [maker, model, focalLength, aperture ?? '', distance ?? ''].join('\u0000');

  const cached = detectionCache.get(cacheKey);
  if (cached) return cached;

  const request = invoke<DetectedLensProfile | null>('autodetect_lens_profile', {
    aperture,
    distance,
    focalLength,
    maker,
    model,
  }).catch((error) => {
    if (detectionCache.get(cacheKey) === request) {
      detectionCache.delete(cacheKey);
    }
    throw error;
  });

  detectionCache.set(cacheKey, request);
  if (detectionCache.size > MAX_DETECTION_CACHE_ENTRIES) {
    const oldestKey = detectionCache.keys().next().value;
    if (oldestKey && oldestKey !== cacheKey) detectionCache.delete(oldestKey);
  }
  return request;
};
