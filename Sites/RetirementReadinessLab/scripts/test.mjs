import {readdirSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
if(Number(process.versions.node.split('.')[0])<22){
  console.error('Sites tests require Node.js 22 or newer.');process.exit(1);
}
const tests=readdirSync(new URL('../tests/',import.meta.url)).filter(name=>name.endsWith('.test.js')).sort().map(name=>`tests/${name}`);
const result=spawnSync(process.execPath,['--test',...tests],{cwd:root,stdio:'inherit'});
if(result.error)console.error(result.error.message);
process.exit(result.status??1);
