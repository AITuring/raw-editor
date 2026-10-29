import fs from 'node:fs';
import path from 'node:path';
import { readRustModule } from './rust-source.mjs';

// Source contracts read a module as one text, so their checks follow the
// implementation instead of a single-file layout:
// - Rust: `name.rs` plus its `name/` child modules (see rust-source.mjs);
// - TypeScript: `Name.tsx` (or `.ts`) plus every `.ts`/`.tsx` part under the
//   sibling `Name/` directory, in sorted order;
// - CSS: the stylesheet with each relative `@import` inlined in place, right
//   after the `@import` line, so rules keep their original order.

const TS_EXTENSIONS = new Set(['.ts', '.tsx']);

const listFiles = (directory, extensions) =>
  fs
    .readdirSync(directory, { withFileTypes: true })
    .flatMap((entry) => {
      const entryPath = path.join(directory, entry.name);
      if (entry.isDirectory()) return listFiles(entryPath, extensions);
      return extensions.has(path.extname(entry.name)) ? [entryPath] : [];
    })
    .sort();

/**
 * Parts of a split TypeScript module as repository-relative paths, or an empty
 * list when `Name/` does not exist. The directory name must match the file
 * stem exactly, so a differently cased sibling directory on a case-insensitive
 * file system (`Editor.tsx` next to `editor/`) is not mistaken for its parts.
 */
export const tsModuleParts = (repoRoot, relativePath) => {
  const extension = path.extname(relativePath);
  if (!TS_EXTENSIONS.has(extension)) return [];
  const directory = path.join(repoRoot, relativePath.slice(0, -extension.length));
  const parent = path.dirname(directory);
  if (!fs.existsSync(parent) || !fs.readdirSync(parent).includes(path.basename(directory))) return [];
  if (!fs.statSync(directory).isDirectory()) return [];
  return listFiles(directory, TS_EXTENSIONS).map((file) => path.relative(repoRoot, file).split(path.sep).join('/'));
};

const RELATIVE_CSS_IMPORT = /^@import\s+(['"])(\.{1,2}\/[^'"]+\.css)\1[^\n]*$/gm;

const readStylesheet = (repoRoot, relativePath, seen) => {
  const text = fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');
  return text.replace(RELATIVE_CSS_IMPORT, (line, _quote, target) => {
    const imported = path.posix.normalize(path.posix.join(path.posix.dirname(relativePath), target));
    if (seen.has(imported)) return line;
    seen.add(imported);
    return `${line}\n${readStylesheet(repoRoot, imported, seen)}`;
  });
};

/** Source text of a module, following the layout rules above. */
export const readSourceModule = (repoRoot, relativePath) => {
  if (relativePath.endsWith('.rs')) return readRustModule(repoRoot, relativePath);
  if (relativePath.endsWith('.css')) return readStylesheet(repoRoot, relativePath, new Set([relativePath]));
  return [relativePath, ...tsModuleParts(repoRoot, relativePath)]
    .map((file) => fs.readFileSync(path.join(repoRoot, file), 'utf8'))
    .join('\n');
};
