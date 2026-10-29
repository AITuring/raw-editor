export function getHistogramPath(data: Array<any>) {
  if (!data || data.length === 0) return '';
  const maxVal = Math.max(...data);
  if (maxVal === 0) return '';

  const pathData = data
    .map((value: number, index: number) => `${(index / 255) * 255},${255 - (value / maxVal) * 255}`)
    .join(' ');

  return `M0,255 L${pathData} L255,255 Z`;
}

export function getZeroHistogramPath(data: Array<any>) {
  if (!data || data.length === 0) return '';
  const pathData = data.map((_, index: number) => `${(index / 255) * 255},255`).join(' ');
  return `M0,255 L${pathData} L255,255 Z`;
}
