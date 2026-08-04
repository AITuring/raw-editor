export function createImageObjectUrl(buffer: ArrayBuffer, mimeType: string): string | null {
  if (!buffer || buffer.byteLength === 0) return null;
  return URL.createObjectURL(new Blob([buffer], { type: mimeType }));
}

export function revokeImageObjectUrl(url: string | null | undefined): void {
  if (url?.startsWith('blob:')) URL.revokeObjectURL(url);
}

export function revokeImageObjectUrlLater(url: string | null | undefined, delay = 250): void {
  if (!url?.startsWith('blob:')) return;
  setTimeout(() => URL.revokeObjectURL(url), delay);
}
