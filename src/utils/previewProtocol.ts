const WGPU_RENDER_MAGIC = 'WGPU_RENDER';
const PREVIEW_PATCH_MAGIC = 'RAWROI01';
const PREVIEW_PATCH_HEADER_BYTES = 32;

export type PreviewResponse =
  | { kind: 'wgpu' }
  | { kind: 'full'; imageBuffer: ArrayBuffer }
  | {
      kind: 'patch';
      imageBuffer: ArrayBuffer;
      normX: number;
      normY: number;
      normW: number;
      normH: number;
    };

const decoder = new TextDecoder();

export function parsePreviewResponse(buffer: ArrayBuffer): PreviewResponse {
  const wgpuPrefix = decoder.decode(buffer.slice(0, WGPU_RENDER_MAGIC.length));
  if (wgpuPrefix === WGPU_RENDER_MAGIC) {
    return { kind: 'wgpu' };
  }

  const patchPrefix = decoder.decode(buffer.slice(0, PREVIEW_PATCH_MAGIC.length));
  if (patchPrefix !== PREVIEW_PATCH_MAGIC) {
    return { kind: 'full', imageBuffer: buffer };
  }

  if (buffer.byteLength <= PREVIEW_PATCH_HEADER_BYTES) {
    throw new Error('Invalid preview patch: missing JPEG payload');
  }

  const view = new DataView(buffer);
  const patchX = view.getUint32(8, true);
  const patchY = view.getUint32(12, true);
  const patchW = view.getUint32(16, true);
  const patchH = view.getUint32(20, true);
  const fullW = view.getUint32(24, true);
  const fullH = view.getUint32(28, true);

  if (
    patchW === 0 ||
    patchH === 0 ||
    fullW === 0 ||
    fullH === 0 ||
    patchX + patchW > fullW ||
    patchY + patchH > fullH
  ) {
    throw new Error('Invalid preview patch geometry');
  }

  return {
    kind: 'patch',
    imageBuffer: buffer.slice(PREVIEW_PATCH_HEADER_BYTES),
    normX: patchX / fullW,
    normY: patchY / fullH,
    normW: patchW / fullW,
    normH: patchH / fullH,
  };
}
