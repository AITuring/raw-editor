const coverImageModules = import.meta.glob('../assets/cover/optimized/*.webp', {
  eager: true,
  import: 'default',
  query: '?url',
}) as Record<string, string>;

export const COVER_IMAGES = Object.entries(coverImageModules)
  .sort(([firstPath], [secondPath]) => firstPath.localeCompare(secondPath, undefined, { numeric: true }))
  .map(([, imageUrl]) => imageUrl);

export const COVER_ROTATION_INTERVAL_MS = 8_000;
