import fs from 'node:fs';
import path from 'node:path';
import { auditReport, fileIdentity } from './support/station-scale-audit.mjs';

const [reportPath, alignmentPath, outputPath] = process.argv.slice(2);
if (!reportPath || !alignmentPath || !outputPath) {
  throw new Error('Usage: node tests/focus-stack-scale-audit.mjs REPORT SOURCE_ALIGNMENT NEW_OUTPUT');
}
if ([reportPath,alignmentPath].some(p => path.resolve(p) === path.resolve(outputPath))) {
  throw new Error('Refusing to overwrite audit input');
}
const inputs = [reportPath,alignmentPath].map(fileIdentity);
const result = auditReport(JSON.parse(fs.readFileSync(reportPath)), JSON.parse(fs.readFileSync(alignmentPath)));
fs.writeFileSync(outputPath, JSON.stringify({ ...result, inputs },null,2)+'\n', { flag: 'wx' });
console.log(JSON.stringify({ diagnostic_written: outputPath, production: result.production }));
