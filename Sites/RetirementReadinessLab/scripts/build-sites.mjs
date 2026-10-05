import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { build } from 'esbuild';

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
// Keep every client module (including worker imports) in the same immutable
// release directory. Worker URLs resolve against import.meta.url in app.js.
const client=path.join(dist,'client');
const modules=(await readdir(client)).filter(name=>name.endsWith('.js')).sort();
const hash=createHash('sha256');
for(const name of modules){hash.update(name);hash.update(await readFile(path.join(client,name)));}
const release=hash.digest('hex').slice(0,20),assetDir=path.join(client,'assets',release);
await mkdir(assetDir,{recursive:true});
for(const name of modules)await cp(path.join(client,name),path.join(assetDir,name));
const index=await readFile(path.join(client,'index.html'),'utf8');
await writeFile(path.join(client,'index.html'),index.replace(/src="\.\/app\.js(?:\?[^\"]*)?"/,`src="./assets/${release}/app.js"`));
const workerSource = await readFile(path.join(root, 'worker/index.js'), 'utf8');
const adminHtml = await readFile(path.join(root, 'dist/admin.html'), 'utf8');
const verificationHtml = await readFile(path.join(root,'worker/mcp-verification.html'),'utf8');
if (!workerSource.includes("'__ADMIN_HTML__'")) throw new Error('Missing admin page placeholder in worker/index.js');
// A replacer function inserts the page literally. A replacement string would
// expand $&, $' and $$ if the page ever contained them.
if (!workerSource.includes("'__MCP_VERIFICATION_HTML__'")) throw new Error('Missing owner verification placeholder');
await build({ stdin: { contents: workerSource.replace("'__ADMIN_HTML__'", () => JSON.stringify(adminHtml)).replace("'__MCP_VERIFICATION_HTML__'",()=>JSON.stringify(verificationHtml)), resolveDir: path.join(root, 'worker'), sourcefile: 'index.js', loader: 'js' },
  outfile: path.join(dist, 'server/index.js'), bundle: true, format: 'esm', platform: 'browser', conditions: ['workerd', 'browser'], target: 'es2022', minify: true });
await cp(path.join(root, 'drizzle'), path.join(dist, '.openai/drizzle'), { recursive: true });
await cp(path.join(root, '.openai/hosting.json'), path.join(output, '.openai/hosting.json'));
await cp(path.join(root, '.openai/hosting.json'), path.join(dist, '.openai/hosting.json'));
await writeFile(path.join(dist, 'server/wrangler.json'), JSON.stringify({
  main: 'index.js',
  compatibility_date: '2026-09-01',
  limits: { cpu_ms: 30000 },
  assets: { directory: '../client', binding: 'ASSETS', run_worker_first: true },
}, null, 2) + '\n');
console.log(`Prepared Sites Worker build in ${output}`);
