import { type ImageDataLike } from '../../../utils/styleTransfer';
import { MAX_PREVIEW_EDGE } from './constants';
import type { ImageDimensions } from './types';

export const readImageElement = (url: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The selected image could not be decoded.'));
    image.src = url;
  });

export const getFileName = (path: string) => path.split(/[\\/]/).pop() || path;

export const getOutputName = (name: string) => {
  const withoutExtension = name.replace(/\.[^.]+$/, '');
  return `${withoutExtension}_styled.jpg`;
};

export const orientDimensionsLikePreview = (
  dimensions: ImageDimensions | null,
  previewWidth: number,
  previewHeight: number,
): ImageDimensions => {
  if (!dimensions || dimensions.width <= 0 || dimensions.height <= 0) {
    return { width: previewWidth, height: previewHeight };
  }

  const sourceIsLandscape = dimensions.width >= dimensions.height;
  const previewIsLandscape = previewWidth >= previewHeight;
  return sourceIsLandscape === previewIsLandscape ? dimensions : { width: dimensions.height, height: dimensions.width };
};

export const makeImageData = (image: HTMLImageElement): ImageDataLike => {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const scale = Math.min(1, MAX_PREVIEW_EDGE / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Canvas preview is unavailable in this environment.');
  context.drawImage(image, 0, 0, width, height);
  return context.getImageData(0, 0, width, height);
};

export const toNativeImageData = (source: ImageDataLike): ImageData =>
  source instanceof ImageData ? source : new ImageData(new Uint8ClampedArray(source.data), source.width, source.height);
