import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const manifestPath = path.join(here, "acceptance", "focus-stack-datasets.json");
const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const requireData = process.argv.includes("--require-data");
const hashInputs = process.argv.includes("--hash-inputs");
const root = path.resolve(
  process.env[manifest.rootEnvironment] ?? manifest.defaultRoot,
);

function fail(message) {
  console.error(`focus-stack-datasets: ${message}`);
  process.exitCode = 1;
}

function patternToRegExp(pattern) {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${escaped.replaceAll("*", ".*")}$`, "i");
}

function isExcluded(name, patterns) {
  return patterns.some((pattern) => patternToRegExp(pattern).test(name));
}

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function sourceDigest(directory, files) {
  let bytes = 0;
  const records = files.map((name) => {
    const filePath = path.join(directory, name);
    bytes += fs.statSync(filePath).size;
    return `${name}\0${sha256(filePath)}\n`;
  });
  return { bytes, sha256: crypto.createHash("sha256").update(records.join(""), "utf8").digest("hex") };
}

function sipsDimensions(filePath) {
  const result = spawnSync("sips", ["-g", "pixelWidth", "-g", "pixelHeight", filePath], {
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`sips could not inspect ${filePath}: ${result.stderr || "unknown error"}`);
  }
  const width = Number(result.stdout.match(/pixelWidth:\s*(\d+)/)?.[1]);
  const height = Number(result.stdout.match(/pixelHeight:\s*(\d+)/)?.[1]);
  if (!Number.isInteger(width) || !Number.isInteger(height)) {
    throw new Error(`sips returned no dimensions for ${filePath}`);
  }
  return { width, height };
}

if (!fs.existsSync(root)) {
  if (requireData) {
    fail(`dataset root does not exist: ${root}`);
  } else {
    console.log(`focus-stack-datasets: skipped; set ${manifest.rootEnvironment} to run local fixtures`);
  }
  process.exit(process.exitCode ?? 0);
}

const summaries = [];
for (const dataset of manifest.datasets) {
  const directory = path.join(root, dataset.directory);
  if (!fs.existsSync(directory) || !fs.statSync(directory).isDirectory()) {
    fail(`${dataset.id}: directory does not exist: ${directory}`);
    continue;
  }

  const files = fs
    .readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .filter((name) => dataset.sourceExtensions.includes(path.extname(name).toLowerCase()))
    .filter((name) => !isExcluded(name, dataset.excludePatterns));
  files.sort((left, right) => left.localeCompare(right, "en"));

  if (files.length !== dataset.expectedSourceCount) {
    fail(`${dataset.id}: expected ${dataset.expectedSourceCount} inputs, found ${files.length}`);
  }

  // Frozen regression subsets are the first N inputs in code-unit order, the
  // same order tests/focus-stack-regression.mjs passes to the harness.
  for (const subset of dataset.regressionSubsets ?? []) {
    if (
      typeof subset.id !== "string" ||
      !Number.isInteger(subset.first) ||
      subset.first < 2 ||
      subset.first > dataset.expectedSourceCount
    ) {
      fail(`${dataset.id}: invalid regression subset ${JSON.stringify(subset)}`);
    }
  }

  if (hashInputs && dataset.sourceAggregateSha256) {
    const digest = sourceDigest(directory, files);
    if (digest.bytes !== dataset.sourceBytes || digest.sha256 !== dataset.sourceAggregateSha256) {
      fail(
        `${dataset.id}: source manifest changed (bytes=${digest.bytes}, sha256=${digest.sha256})`,
      );
    }
  }

  if (dataset.referenceOutput) {
    const referencePath = path.join(directory, dataset.referenceOutput);
    if (!fs.existsSync(referencePath)) {
      fail(`${dataset.id}: reference output is missing: ${referencePath}`);
    } else {
      if (dataset.referenceSha256) {
        const stat = fs.statSync(referencePath);
        const digest = sha256(referencePath);
        const dimensions = sipsDimensions(referencePath);
        if (stat.size !== dataset.referenceBytes) {
          fail(`${dataset.id}: reference byte size changed (${stat.size} != ${dataset.referenceBytes})`);
        }
        if (digest !== dataset.referenceSha256) {
          fail(`${dataset.id}: reference SHA-256 changed (${digest})`);
        }
        if (
          dimensions.width !== dataset.referenceDimensions.width ||
          dimensions.height !== dataset.referenceDimensions.height
        ) {
          fail(
            `${dataset.id}: reference dimensions changed (${dimensions.width}x${dimensions.height})`,
          );
        }
        for (const alias of dataset.referenceAliases ?? []) {
          const aliasPath = path.join(directory, alias);
          if (!fs.existsSync(aliasPath) || sha256(aliasPath) !== dataset.referenceSha256) {
            fail(`${dataset.id}: reference alias changed or is missing: ${aliasPath}`);
          }
        }
      }
    }
  }

  summaries.push({ id: dataset.id, label: dataset.label, inputs: files.length });
}

if (process.exitCode) {
  process.exit(process.exitCode);
}

console.log(JSON.stringify({ root, datasets: summaries }, null, 2));
