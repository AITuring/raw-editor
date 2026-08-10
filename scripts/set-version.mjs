import { readFileSync, writeFileSync } from 'node:fs';

const version = process.argv[2];
const stableVersionPattern = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;

if (!version || !stableVersionPattern.test(version)) {
  console.error('Usage: npm run version:set -- <major.minor.patch>');
  process.exit(1);
}

function replaceJsonVersion(path) {
  const source = readFileSync(path, 'utf8');
  const pattern = /(^\s*"version"\s*:\s*")[^"]+("\s*,?\s*$)/m;

  if (!pattern.test(source)) {
    throw new Error(`Could not find a version field in ${path}`);
  }

  writeFileSync(path, source.replace(pattern, `$1${version}$2`));
}

replaceJsonVersion('package.json');

const packageLock = JSON.parse(readFileSync('package-lock.json', 'utf8'));
packageLock.version = version;
packageLock.packages[''].version = version;
writeFileSync('package-lock.json', `${JSON.stringify(packageLock, null, 2)}\n`);

replaceJsonVersion('src-tauri/tauri.conf.json');

const cargoTomlPath = 'src-tauri/Cargo.toml';
const cargoToml = readFileSync(cargoTomlPath, 'utf8');
const cargoTomlVersionPattern = /(\[package\][\s\S]*?\nversion\s*=\s*")[^"]+("\s*\n)/;

if (!cargoTomlVersionPattern.test(cargoToml)) {
  throw new Error('Could not update the package version in src-tauri/Cargo.toml');
}

const updatedCargoToml = cargoToml.replace(cargoTomlVersionPattern, `$1${version}$2`);
writeFileSync(cargoTomlPath, updatedCargoToml);

const cargoLockPath = 'src-tauri/Cargo.lock';
const cargoLock = readFileSync(cargoLockPath, 'utf8');
const cargoLockVersionPattern = /(\[\[package\]\]\nname = "raw-editor"\nversion = ")[^"]+("\n)/;

if (!cargoLockVersionPattern.test(cargoLock)) {
  throw new Error('Could not update the raw-editor package version in src-tauri/Cargo.lock');
}

const updatedCargoLock = cargoLock.replace(cargoLockVersionPattern, `$1${version}$2`);
writeFileSync(cargoLockPath, updatedCargoLock);

console.log(`Updated RAW Editor version to ${version}`);
