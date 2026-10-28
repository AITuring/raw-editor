import fs from 'node:fs';
import path from 'node:path';

// Rust modules are split as `name.rs` (the facade) plus a `name/` directory of
// child modules. Source contracts read such a module as one text so their
// checks follow the implementation instead of a single-file layout.

const listRustFiles = (directory) =>
  fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) return listRustFiles(entryPath);
    return entry.name.endsWith('.rs') ? [entryPath] : [];
  });

/**
 * Child module files of a split Rust module, as repository-relative paths in
 * sorted order (`name/a.rs`, `name/a/b.rs`, `name/c.rs`). Returns an empty list
 * when the module has not been split. `mod.rs` declares its children next to
 * itself, so it has no child directory of this kind; a split `lib.rs` keeps its
 * children in `lib/` through `#[path = "lib/<child>.rs"]` declarations.
 */
export const rustChildModuleFiles = (repoRoot, relativePath) => {
  const base = path.basename(relativePath);
  if (!relativePath.endsWith('.rs') || base === 'mod.rs' || base === 'main.rs') return [];
  const directory = path.join(repoRoot, relativePath.slice(0, -'.rs'.length));
  if (!fs.existsSync(directory) || !fs.statSync(directory).isDirectory()) return [];
  return listRustFiles(directory)
    .map((file) => path.relative(repoRoot, file).split(path.sep).join('/'))
    .sort();
};

/** Source of a Rust module: the facade followed by every split child module. */
export const readRustModule = (repoRoot, relativePath) =>
  [relativePath, ...rustChildModuleFiles(repoRoot, relativePath)]
    .map((file) => fs.readFileSync(path.join(repoRoot, file), 'utf8'))
    .join('\n');

/**
 * Whether a child module file (`a/b/c.rs`) is compiled only for tests: its
 * parent (`a/b.rs`) declares it behind `#[cfg(test)]`, the file opens with
 * `#![cfg(test)]`, or its parent is itself a test-only child.
 */
export const isTestOnlyRustChild = (repoRoot, relativePath) => {
  const parent = `${path.posix.dirname(relativePath)}.rs`;
  const parentPath = path.join(repoRoot, parent);
  if (!fs.existsSync(parentPath)) return false;
  if (/^\s*#!\[cfg\(test\)\]/.test(fs.readFileSync(path.join(repoRoot, relativePath), 'utf8'))) return true;
  const stem = path.posix.basename(relativePath, '.rs');
  const declaration = new RegExp(
    `#\\[cfg\\(test\\)\\]\\s*(?:#\\[[^\\]]*\\]\\s*)*(?:pub(?:\\([^)]*\\))?\\s+)?mod\\s+${stem}\\s*;`,
  );
  if (declaration.test(fs.readFileSync(parentPath, 'utf8'))) return true;
  return isTestOnlyRustChild(repoRoot, parent);
};
