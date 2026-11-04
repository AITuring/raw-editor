export interface BrightnessRegion {
  index: number;
  row: number;
  column: number;
  x: number;
  y: number;
  width: number;
  height: number;
  average: number;
  /** The middle sampled luminance, less sensitive to a dark subject or bright highlight. */
  median: number;
  minimum: number;
  maximum: number;
}

export interface BrightnessGrid {
  rows: number;
  columns: number;
  regions: BrightnessRegion[];
  average: number;
  minimum: number;
  maximum: number;
  /** Luminance difference measured immediately across each vertical seam. */
  verticalSeamDeltas?: number[];
  /** Luminance difference measured immediately across each horizontal seam. */
  horizontalSeamDeltas?: number[];
}

const clamp = (value: number, minimum = 0, maximum = 255) => Math.min(maximum, Math.max(minimum, value));

const luminance = (red: number, green: number, blue: number) => red * 0.2126 + green * 0.7152 + blue * 0.0722;

function median(values: number[]) {
  if (values.length === 0) return 0;
  values.sort((left, right) => left - right);
  const middle = Math.floor(values.length / 2);
  return values.length % 2 === 0 ? (values[middle - 1] + values[middle]) / 2 : values[middle];
}

function sampleMedian(imageData: ImageData, x: number, y: number, width: number, height: number, stride: number) {
  const samples: number[] = [];
  const startX = Math.max(0, Math.floor(x));
  const startY = Math.max(0, Math.floor(y));
  const endX = Math.min(imageData.width, Math.max(startX + 1, Math.ceil(x + width)));
  const endY = Math.min(imageData.height, Math.max(startY + 1, Math.ceil(y + height)));
  for (let sampleY = startY; sampleY < endY; sampleY += stride) {
    for (let sampleX = startX; sampleX < endX; sampleX += stride) {
      const offset = (sampleY * imageData.width + sampleX) * 4;
      samples.push(luminance(imageData.data[offset], imageData.data[offset + 1], imageData.data[offset + 2]));
    }
  }
  return median(samples);
}

/**
 * Samples an ImageData into a regular grid. The returned values are in the
 * 0–255 luminance range, making them useful both for display and adjustment.
 */
export function analyzeBrightnessRegions(imageData: ImageData, rows = 4, columns = 6): BrightnessGrid {
  const safeRows = Math.min(imageData.height, Math.max(1, Math.floor(rows)));
  const safeColumns = Math.min(imageData.width, Math.max(1, Math.floor(columns)));
  const sampleStride = Math.max(1, Math.ceil(Math.sqrt((imageData.width * imageData.height) / 300_000)));
  const regions: BrightnessRegion[] = [];
  let globalSum = 0;
  let globalSampleCount = 0;
  let globalMinimum = 255;
  let globalMaximum = 0;

  for (let row = 0; row < safeRows; row += 1) {
    const y = Math.floor((row * imageData.height) / safeRows);
    const nextY = Math.max(y + 1, Math.floor(((row + 1) * imageData.height) / safeRows));
    for (let column = 0; column < safeColumns; column += 1) {
      const x = Math.floor((column * imageData.width) / safeColumns);
      const nextX = Math.max(x + 1, Math.floor(((column + 1) * imageData.width) / safeColumns));
      let sum = 0;
      let minimum = 255;
      let maximum = 0;
      let count = 0;
      const samples: number[] = [];

      for (let pixelY = y; pixelY < nextY; pixelY += sampleStride) {
        for (let pixelX = x; pixelX < nextX; pixelX += sampleStride) {
          const offset = (pixelY * imageData.width + pixelX) * 4;
          const value = luminance(imageData.data[offset], imageData.data[offset + 1], imageData.data[offset + 2]);
          sum += value;
          samples.push(value);
          minimum = Math.min(minimum, value);
          maximum = Math.max(maximum, value);
          globalSum += value;
          globalSampleCount += 1;
          globalMinimum = Math.min(globalMinimum, value);
          globalMaximum = Math.max(globalMaximum, value);
          count += 1;
        }
      }

      regions.push({
        index: row * safeColumns + column,
        row,
        column,
        x,
        y,
        width: nextX - x,
        height: nextY - y,
        average: count > 0 ? sum / count : 0,
        median: count > 0 ? median(samples) : 0,
        minimum: count > 0 ? minimum : 0,
        maximum: count > 0 ? maximum : 0,
      });
    }
  }

  const verticalSeamDeltas: number[] = [];
  const horizontalSeamDeltas: number[] = [];
  const verticalInset = Math.max(1, Math.round(imageData.height / safeRows * 0.08));
  const horizontalInset = Math.max(1, Math.round(imageData.width / safeColumns * 0.08));
  for (let row = 0; row < safeRows; row += 1) {
    const regionY = Math.floor((row * imageData.height) / safeRows);
    const regionNextY = Math.max(regionY + 1, Math.floor(((row + 1) * imageData.height) / safeRows));
    const stripY = regionY + verticalInset;
    const stripHeight = Math.max(1, regionNextY - regionY - verticalInset * 2);
    for (let column = 0; column < safeColumns - 1; column += 1) {
      const boundaryX = Math.floor(((column + 1) * imageData.width) / safeColumns);
      const regionWidth = Math.max(1, Math.floor(imageData.width / safeColumns));
      const stripWidth = Math.max(1, Math.round(regionWidth * 0.06));
      const left = sampleMedian(imageData, boundaryX - stripWidth, stripY, stripWidth, stripHeight, sampleStride);
      const right = sampleMedian(imageData, boundaryX, stripY, stripWidth, stripHeight, sampleStride);
      verticalSeamDeltas.push(clamp(left - right, -100, 100));
    }
  }
  for (let row = 0; row < safeRows - 1; row += 1) {
    const boundaryY = Math.floor(((row + 1) * imageData.height) / safeRows);
    const regionHeight = Math.max(1, Math.floor(imageData.height / safeRows));
    const stripHeight = Math.max(1, Math.round(regionHeight * 0.06));
    const stripWidth = Math.max(1, imageData.width - horizontalInset * 2);
    for (let column = 0; column < safeColumns; column += 1) {
      const regionX = Math.floor((column * imageData.width) / safeColumns);
      const regionNextX = Math.max(regionX + 1, Math.floor(((column + 1) * imageData.width) / safeColumns));
      const x = regionX + Math.min(horizontalInset, Math.max(0, regionNextX - regionX - 1));
      const width = Math.max(1, Math.min(stripWidth, regionNextX - x));
      const top = sampleMedian(imageData, x, boundaryY - stripHeight, width, stripHeight, sampleStride);
      const bottom = sampleMedian(imageData, x, boundaryY, width, stripHeight, sampleStride);
      horizontalSeamDeltas.push(clamp(top - bottom, -100, 100));
    }
  }

  return {
    rows: safeRows,
    columns: safeColumns,
    regions,
    average: globalSampleCount > 0 ? globalSum / globalSampleCount : 0,
    minimum: globalSampleCount > 0 ? globalMinimum : 0,
    maximum: globalSampleCount > 0 ? globalMaximum : 0,
    verticalSeamDeltas,
    horizontalSeamDeltas,
  };
}

/** Return offsets that bring every region towards the same luminance and remove seam steps. */
export function createUniformBrightnessOffsets(grid: BrightnessGrid): number[] {
  const medianWeight = 0.6;
  const metrics = grid.regions.map((region) =>
    region.average * (1 - medianWeight) + (region.median ?? region.average) * medianWeight,
  );
  const target = metrics.length > 0 ? metrics.reduce((sum, metric) => sum + metric, 0) / metrics.length : 0;
  const offsets = metrics.map((metric) => clamp(target - metric, -100, 100));
  const baseMean = offsets.length > 0 ? offsets.reduce((sum, offset) => sum + offset, 0) / offsets.length : 0;
  const verticalSeams = grid.verticalSeamDeltas ?? [];
  const horizontalSeams = grid.horizontalSeamDeltas ?? [];
  if (verticalSeams.length === 0 && horizontalSeams.length === 0) return offsets;

  // Solve pairwise seam constraints with a few relaxed passes. A median strip
  // avoids letting a single dark brush stroke dictate an entire region, while
  // the mean-preserving step keeps the global exposure unchanged.
  for (let iteration = 0; iteration < 5; iteration += 1) {
    let verticalIndex = 0;
    for (let row = 0; row < grid.rows; row += 1) {
      for (let column = 0; column < grid.columns - 1; column += 1) {
        const leftIndex = row * grid.columns + column;
        const rightIndex = leftIndex + 1;
        const desired = verticalSeams[verticalIndex++] ?? 0;
        const regionDifference = grid.regions[leftIndex].average - grid.regions[rightIndex].average;
        // A smooth scene gradient produces a similar difference in the
        // boundary strip and in the whole regions. Only inject seam correction
        // when the narrow strip shows a materially stronger step than the
        // regional means; this keeps natural painted gradients intact.
        if (Math.abs(desired) < 12 || Math.abs(desired) < Math.abs(regionDifference) * 1.35) continue;
        const residual = clamp(desired - (offsets[rightIndex] - offsets[leftIndex]), -60, 60);
        const correction = residual * 0.24;
        offsets[leftIndex] -= correction;
        offsets[rightIndex] += correction;
      }
    }
    let horizontalIndex = 0;
    for (let row = 0; row < grid.rows - 1; row += 1) {
      for (let column = 0; column < grid.columns; column += 1) {
        const topIndex = row * grid.columns + column;
        const bottomIndex = topIndex + grid.columns;
        const desired = horizontalSeams[horizontalIndex++] ?? 0;
        const regionDifference = grid.regions[topIndex].average - grid.regions[bottomIndex].average;
        if (Math.abs(desired) < 12 || Math.abs(desired) < Math.abs(regionDifference) * 1.35) continue;
        const residual = clamp(desired - (offsets[bottomIndex] - offsets[topIndex]), -60, 60);
        const correction = residual * 0.24;
        offsets[topIndex] -= correction;
        offsets[bottomIndex] += correction;
      }
    }
    const mean = offsets.reduce((sum, offset) => sum + offset, 0) / offsets.length;
    const correction = mean - baseMean;
    for (let index = 0; index < offsets.length; index += 1) {
      offsets[index] = clamp(offsets[index] - correction, -100, 100);
    }
  }
  return offsets;
}

interface AxisWeight {
  index: number;
  weight: number;
}

/** Return a small normalized Gaussian kernel around the nearest region centres. */
function gaussianAxisWeights(position: number, count: number, sigma = 0.65): AxisWeight[] {
  if (count <= 0) return [];
  const centrePosition = Math.max(0, Math.min(count - 1, position * count - 0.5));
  const weights: AxisWeight[] = [];
  let totalWeight = 0;
  // Keep the support fixed: moving a truncated kernel at cell centres creates
  // a small but visible jump even though every individual weight is smooth.
  for (let index = -3; index < count + 3; index += 1) {
    const distance = centrePosition - index;
    const weight = Math.exp(-(distance * distance) / (2 * sigma * sigma));
    weights.push({ index: clamp(index, 0, count - 1), weight });
    totalWeight += weight;
  }
  return weights.map(({ index, weight }) => ({ index, weight: weight / totalWeight }));
}

/** Precompute separable fields so exports never rebuild seam kernels per pixel. */
function correctionFields(width: number, height: number, grid: BrightnessGrid, offsets: number[]) {
  const horizontal = Array.from({ length: width }, (_, x) => gaussianAxisWeights((x + 0.5) / width, grid.columns));
  const vertical = Array.from({ length: height }, (_, y) => gaussianAxisWeights((y + 0.5) / height, grid.rows));
  const xField = new Float64Array(width * grid.rows);
  const yField = new Float64Array(height * grid.columns);
  // Restore the complement of the whole blurred step. A narrow correction
  // band hides its centre pixel but leaves bright/dark halos on either side.
  for (let x = 0; x < width; x += 1) {
    const scaledX = (x + 0.5) / width * grid.columns;
    for (let row = 0; row < grid.rows; row += 1) {
      let value = horizontal[x].reduce((sum, item) => sum + (offsets[row * grid.columns + item.index] ?? 0) * item.weight, 0);
      for (let boundary = 1; boundary < grid.columns; boundary += 1) {
        const seam = grid.verticalSeamDeltas?.[row * (grid.columns - 1) + boundary - 1] ?? 0;
        if (Math.abs(seam) < 18) continue;
        const blurredStep = horizontal[x].reduce((sum, item) => sum + (item.index >= boundary ? item.weight : 0), 0);
        const left = row * grid.columns + boundary - 1;
        value += ((offsets[left + 1] ?? 0) - (offsets[left] ?? 0)) * ((scaledX >= boundary ? 1 : 0) - blurredStep);
      }
      xField[x * grid.rows + row] = value;
    }
  }
  for (let y = 0; y < height; y += 1) {
    const scaledY = (y + 0.5) / height * grid.rows;
    for (let column = 0; column < grid.columns; column += 1) {
      let value = 0;
      for (let boundary = 1; boundary < grid.rows; boundary += 1) {
        const seam = grid.horizontalSeamDeltas?.[(boundary - 1) * grid.columns + column] ?? 0;
        if (Math.abs(seam) < 18) continue;
        const blurredStep = vertical[y].reduce((sum, item) => sum + (item.index >= boundary ? item.weight : 0), 0);
        const top = (boundary - 1) * grid.columns + column;
        value += ((offsets[top + grid.columns] ?? 0) - (offsets[top] ?? 0)) * ((scaledY >= boundary ? 1 : 0) - blurredStep);
      }
      yField[y * grid.columns + column] = value;
    }
  }
  return { horizontal, vertical, xField, yField };
}

/** Apply a smoothly interpolated luminance offset field without changing alpha. */
export function applyBrightnessOffsets(source: ImageData, grid: BrightnessGrid, offsets: number[]): ImageData {
  // The initial preview uses an all-zero adjustment field. Keep the source
  // buffer in that state instead of walking every pixel only to add zero.
  if (offsets.every((offset) => offset === 0)) return source;

  const output = new ImageData(new Uint8ClampedArray(source.data), source.width, source.height);

  const { horizontal, vertical, xField, yField } = correctionFields(source.width, source.height, grid, offsets);

  for (let y = 0; y < source.height; y += 1) {
    for (let x = 0; x < source.width; x += 1) {
      let offset = 0;
      for (const item of vertical[y]) offset += xField[x * grid.rows + item.index] * item.weight;
      for (const item of horizontal[x]) offset += yField[y * grid.columns + item.index] * item.weight;
      if (offset === 0) continue;
      const pixelOffset = (y * source.width + x) * 4;
      const delta = offset;
      output.data[pixelOffset] = clamp(output.data[pixelOffset] + delta);
      output.data[pixelOffset + 1] = clamp(output.data[pixelOffset + 1] + delta);
      output.data[pixelOffset + 2] = clamp(output.data[pixelOffset + 2] + delta);
    }
  }

  return output;
}
