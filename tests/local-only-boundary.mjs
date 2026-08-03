import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const sourceExtensions = new Set(['.js', '.mjs', '.ts', '.tsx', '.rs']);

async function collectSourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectSourceFiles(entryPath)));
    } else if (sourceExtensions.has(path.extname(entry.name))) {
      files.push(entryPath);
    }
  }
  return files;
}

const sourceFiles = [
  ...(await collectSourceFiles(path.join(projectRoot, 'src'))),
  ...(await collectSourceFiles(path.join(projectRoot, 'src-tauri', 'src'))),
];
const boundaryFiles = [
  ...sourceFiles,
  path.join(projectRoot, 'package.json'),
  path.join(projectRoot, 'package-lock.json'),
  path.join(projectRoot, 'src-tauri', 'Cargo.toml'),
  path.join(projectRoot, 'src-tauri', 'Cargo.lock'),
  path.join(projectRoot, 'src-tauri', 'capabilities', 'default.json'),
];

const forbiddenProductPatterns = [
  ['RapidRAW cloud endpoint', /getrapidraw\.com/i],
  ['online community preset endpoint', /RapidRAW-Presets/i],
  ['community page', /\bCommunityPage\b/],
  ['community fetch command', /\bfetch_community_presets\b/],
  ['community preview command', /\bgenerate_all_community_previews\b/],
  ['community save command', /\bsave_community_preset\b/],
  ['remote AI connector module', /\bai_connector\b/],
  ['remote AI connector event', /ai-connector-status/i],
  ['account provider', /@clerk\//i],
  ['shell plugin', /(?:@tauri-apps\/plugin-shell|tauri-plugin-shell|shell:default)/i],
];

const failures = [];
for (const file of boundaryFiles) {
  const content = await readFile(file, 'utf8');
  const relativePath = path.relative(projectRoot, file);
  for (const [description, pattern] of forbiddenProductPatterns) {
    if (pattern.test(content)) failures.push(`${relativePath}: ${description}`);
  }

  if (/^src[/\\].*\.(?:js|mjs|ts|tsx)$/.test(relativePath)) {
    if (/\b(?:fetch|WebSocket|EventSource|XMLHttpRequest)\s*\(/.test(content)) {
      failures.push(`${relativePath}: frontend network API`);
    }
    for (const match of content.matchAll(/https?:\/\/[^"'`\s>]+/g)) {
      if (match[0] !== 'http://www.w3.org/2000/svg') {
        failures.push(`${relativePath}: external runtime URL ${match[0]}`);
      }
    }
  }

  if (relativePath.endsWith('.rs') && /(?:reqwest::|tokio::net|std::net::(?:TcpStream|UdpSocket))/.test(content)) {
    if (relativePath !== path.join('src-tauri', 'src', 'ai_processing.rs')) {
      failures.push(`${relativePath}: network client outside the public model downloader`);
    }
  }
}

const buildScript = await readFile(path.join(projectRoot, 'src-tauri', 'build.rs'), 'utf8');
for (const [relativePath, content] of [
  [
    path.join('src-tauri', 'src', 'ai_processing.rs'),
    await readFile(path.join(projectRoot, 'src-tauri', 'src', 'ai_processing.rs'), 'utf8'),
  ],
  [path.join('src-tauri', 'build.rs'), buildScript],
]) {
  const urls = [...content.matchAll(/https:\/\/[^"\s]+/g)].map((match) => match[0]);
  if (urls.length === 0) failures.push(`${relativePath}: expected an explicit public asset URL`);
  for (const url of urls) {
    if (!url.startsWith('https://huggingface.co/CyberTimon/RapidRAW-Models/')) {
      failures.push(`${relativePath}: non-allowlisted download host ${url}`);
    }
  }
}

if (failures.length > 0) {
  console.error('Local-only product boundary violated:');
  for (const failure of [...new Set(failures)].sort()) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log(`Validated the local-only product boundary across ${boundaryFiles.length} files.`);
}
