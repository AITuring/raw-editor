import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

class TestImageData {
  constructor(dataOrWidth, widthOrHeight, height) {
    if (typeof dataOrWidth === 'number') {
      this.width = dataOrWidth;
      this.height = widthOrHeight;
      this.data = new Uint8ClampedArray(this.width * this.height * 4);
    } else {
      this.data = dataOrWidth;
      this.width = widthOrHeight;
      this.height = height;
    }
  }
}
globalThis.ImageData = TestImageData;

const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/brightnessMap.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const moduleSource = Buffer.from(bundled.outputFiles[0].contents).toString('base64');
const { analyzeBrightnessRegions, createUniformBrightnessOffsets, applyBrightnessOffsets } = await import(
  `data:text/javascript;base64,${moduleSource}`
);

const makeImage = (width, height, pixel) => {
  const image = new TestImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const [red, green, blue, alpha] = pixel(x, y);
      image.data[index] = red;
      image.data[index + 1] = green;
      image.data[index + 2] = blue;
      image.data[index + 3] = alpha;
    }
  }
  return image;
};

const luminance = (red, green, blue) => red * 0.2126 + green * 0.7152 + blue * 0.0722;
const regionMeans = (image, rows, columns) => {
  const values = [];
  for (let row = 0; row < rows; row += 1) {
    const y0 = Math.floor((row * image.height) / rows);
    const y1 = Math.max(y0 + 1, Math.floor(((row + 1) * image.height) / rows));
    for (let column = 0; column < columns; column += 1) {
      const x0 = Math.floor((column * image.width) / columns);
      const x1 = Math.max(x0 + 1, Math.floor(((column + 1) * image.width) / columns));
      let sum = 0;
      let count = 0;
      for (let y = y0; y < y1; y += 1) {
        for (let x = x0; x < x1; x += 1) {
          const index = (y * image.width + x) * 4;
          sum += luminance(image.data[index], image.data[index + 1], image.data[index + 2]);
          count += 1;
        }
      }
      values.push(sum / count);
    }
  }
  return values;
};

// A smooth horizontal brightness field should move towards a common regional mean.
const gradient = makeImage(24, 12, (x, y) => {
  const value = 45 + x * 5 + y * 0.4;
  return [value, value, value, 17 + ((x + y) % 230)];
});
const gradientGrid = analyzeBrightnessRegions(gradient, 3, 4);
assert.equal(gradientGrid.regions.length, 12);
assert.equal(gradientGrid.minimum, 45);
assert.ok(gradientGrid.maximum > gradientGrid.minimum);
const gradientBefore = regionMeans(gradient, 3, 4);
const gradientOffsets = createUniformBrightnessOffsets(gradientGrid);
assert.ok(gradientOffsets.some((offset) => offset < 0));
assert.ok(gradientOffsets.some((offset) => offset > 0));
const corrected = applyBrightnessOffsets(gradient, gradientGrid, gradientOffsets);
const gradientAfter = regionMeans(corrected, 3, 4);
const beforeSpread = Math.max(...gradientBefore) - Math.min(...gradientBefore);
const afterSpread = Math.max(...gradientAfter) - Math.min(...gradientAfter);
assert.ok(beforeSpread > 40, `gradient must have visible regional spread: ${beforeSpread}`);
assert.ok(
  afterSpread < beforeSpread * 0.2,
  `smooth offsets should strongly reduce regional spread: ${beforeSpread} -> ${afterSpread}`,
);
for (let index = 3; index < gradient.data.length; index += 4) {
  assert.equal(corrected.data[index], gradient.data[index], 'brightness correction must preserve alpha');
}

// Exposure steps that happen exactly at a region boundary should be corrected
// at the boundary itself, instead of leaving a half-corrected seam band.
const stepped = makeImage(240, 40, (x) => {
  const value = Math.floor(x / 40) % 2 === 0 ? 70 : 130;
  return [value, value, value, 255];
});
const steppedGrid = analyzeBrightnessRegions(stepped, 4, 6);
const steppedOutput = applyBrightnessOffsets(stepped, steppedGrid, createUniformBrightnessOffsets(steppedGrid));
for (let column = 1; column < 6; column += 1) {
  let seamDelta = 0;
  for (let y = 0; y < stepped.height; y += 1) {
    const left = (y * stepped.width + column * 40 - 1) * 4;
    const right = (y * stepped.width + column * 40) * 4;
    seamDelta += Math.abs(steppedOutput.data[left] - steppedOutput.data[right]);
  }
  assert.ok(seamDelta / stepped.height <= 1, `boundary seam should be corrected: column ${column}`);
}

// Removing only the single boundary pixel is insufficient: no bright/dark
// halo may remain on either side of an otherwise uniform exposure step.
const stepValues = [];
for (let x = 45; x < 195; x += 1) stepValues.push(steppedOutput.data[(20 * 240 + x) * 4]);
assert.ok(Math.max(...stepValues) - Math.min(...stepValues) <= 3,
  `corrected steps must not leave bands: ${Math.min(...stepValues)}..${Math.max(...stepValues)}`);

// The same contract must hold for horizontal steps and crossing additive
// exposure errors, not only the vertical fixture that originally hid halos.
const crossedSteps = makeImage(240, 160, (x, y) => {
  const value = 110 + (Math.floor(x / 40) % 2 ? 25 : -25) + (Math.floor(y / 40) % 2 ? 25 : -25);
  return [value, value, value, 255];
});
const crossedGrid = analyzeBrightnessRegions(crossedSteps, 4, 6);
const crossedResult = applyBrightnessOffsets(crossedSteps, crossedGrid, createUniformBrightnessOffsets(crossedGrid));
for (let index = 0; index < crossedResult.data.length; index += 4) {
  assert.ok(Math.abs(crossedResult.data[index] - 110) <= 1, 'crossing exposure steps must not leave halos or double correction');
}

// Equal means can still hide a perceptual brightness mismatch when one region
// has a broad dark population and the other has a broad light population.
const distributionImage = makeImage(20, 8, (x) => {
  const value = x < 10 ? (x < 5 ? 60 : 140) : x < 18 ? 80 : 180;
  return [value, value, value, 255];
});
const distributionGrid = analyzeBrightnessRegions(distributionImage, 1, 2);
assert.ok(Math.abs(distributionGrid.regions[0].average - distributionGrid.regions[1].average) < 1);
assert.ok(distributionGrid.regions[0].median !== distributionGrid.regions[1].median);
const distributionOffsets = createUniformBrightnessOffsets(distributionGrid);
assert.ok(
  Math.abs(distributionOffsets[0] - distributionOffsets[1]) > 10,
  'median-aware offsets should correct equal-mean regions with different distributions',
);

// A uniform field should produce zero offsets and byte-for-byte identical output.
const uniform = makeImage(11, 7, () => [121, 121, 121, 255]);
const uniformGrid = analyzeBrightnessRegions(uniform, 4, 5);
const uniformOffsets = createUniformBrightnessOffsets(uniformGrid);
assert.ok(uniformOffsets.every((offset) => offset === 0));
const uniformOutput = applyBrightnessOffsets(uniform, uniformGrid, uniformOffsets);
assert.deepEqual([...uniformOutput.data], [...uniform.data]);

// Tiny images and invalid grid sizes still yield a valid one-cell analysis.
const tiny = makeImage(1, 1, () => [10, 20, 30, 99]);
const tinyGrid = analyzeBrightnessRegions(tiny, 0, -2);
assert.deepEqual({ rows: tinyGrid.rows, columns: tinyGrid.columns }, { rows: 1, columns: 1 });
assert.equal(tinyGrid.regions[0].average, luminance(10, 20, 30));

// A vertical correction must not acquire a horizontal step at a row boundary.
const flat = makeImage(800, 800, () => [128, 128, 128, 255]);
const crossingGrid = analyzeBrightnessRegions(flat, 2, 2);
crossingGrid.verticalSeamDeltas = [60, -60];
crossingGrid.horizontalSeamDeltas = [0, 0];
const crossing = applyBrightnessOffsets(flat, crossingGrid, [30, -30, -30, 30]);
for (const x of [395, 399, 400, 404]) {
  for (let y = 1; y < 800; y += 1) {
    const jump = Math.abs(crossing.data[(y * 800 + x) * 4] - crossing.data[((y - 1) * 800 + x) * 4]);
    assert.ok(jump <= 1, `seam correction must stay continuous along its length: ${x},${y}: ${jump}`);
  }
}
const defaultTiny = analyzeBrightnessRegions(tiny);
assert.equal(defaultTiny.regions.length, 1);
assert.ok(Number.isFinite(defaultTiny.average));

console.log('brightness-map checks passed');
