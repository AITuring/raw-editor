export const getEdgeFadeStyle = (fadeDistancePx: number = 128): React.CSSProperties => ({
  WebkitMaskImage: `
    linear-gradient(to right, transparent, black ${fadeDistancePx}px, black calc(100% - ${fadeDistancePx}px), transparent),
    linear-gradient(to bottom, transparent, black ${fadeDistancePx}px, black calc(100% - ${fadeDistancePx}px), transparent)
  `,
  WebkitMaskComposite: 'source-in',
  maskImage: `
    linear-gradient(to right, transparent, black ${fadeDistancePx}px, black calc(100% - ${fadeDistancePx}px), transparent),
    linear-gradient(to bottom, transparent, black ${fadeDistancePx}px, black calc(100% - ${fadeDistancePx}px), transparent)
  `,
  maskComposite: 'intersect',
});
