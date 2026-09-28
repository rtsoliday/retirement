import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, '.sites-build');
const dist = path.join(output, 'dist');
const hosting = JSON.parse(await readFile(path.join(root, '.openai/hosting.json'), 'utf8'));
if (!hosting.project_id) throw new Error('Missing Sites project ID');
await rm(output, { recursive: true, force: true });
await mkdir(path.join(dist, 'server'), { recursive: true });
await mkdir(path.join(output, '.openai'), { recursive: true });
await mkdir(path.join(dist, '.openai'), { recursive: true });
await cp(path.join(root, 'dist'), path.join(dist, 'client'), { recursive: true });
await rm(path.join(dist, 'client/admin.html'));
const workerSource = await readFile(path.join(root, 'worker/index.js'), 'utf8');
const adminHtml = await readFile(path.join(root, 'dist/admin.html'), 'utf8');
await writeFile(path.join(dist, 'server/index.js'), workerSource.replace("'__ADMIN_HTML__'", JSON.stringify(adminHtml)));
await cp(path.join(root, 'worker/billing.js'), path.join(dist, 'server/billing.js'));
await cp(path.join(root, 'worker/auth.js'), path.join(dist, 'server/auth.js'));
await cp(path.join(root, '.openai/hosting.json'), path.join(output, '.openai/hosting.json'));
await cp(path.join(root, '.openai/hosting.json'), path.join(dist, '.openai/hosting.json'));
await writeFile(path.join(dist, 'server/wrangler.json'), JSON.stringify({
  main: 'index.js',
  compatibility_date: '2026-09-01',
  assets: { directory: '../client', binding: 'ASSETS', run_worker_first: true },
}, null, 2) + '\n');
console.log(`Prepared Sites Worker build in ${output}`);
