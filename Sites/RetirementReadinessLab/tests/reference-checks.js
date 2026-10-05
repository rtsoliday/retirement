import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

export function verifyReference(name){
 const root=new URL(`./reference/${name}/`,import.meta.url),manifest=JSON.parse(readFileSync(new URL('manifest.json',root),'utf8'));
 for(const[file,sha]of Object.entries(manifest.files))assert.equal(createHash('sha256').update(readFileSync(new URL(file,root))).digest('hex'),sha,`Frozen ${name}/${file} changed; preserve the original reference.`);
 return manifest;
}
// Exact comparison on one runtime avoids treating Math implementation changes
// as financial regressions. A failure identifies the first changed field.
export function assertReference(actual,expected,label){
 function visit(a,b,path){
  if(Object.is(a,b))return;
  if(a&&b&&typeof a==='object'&&typeof b==='object'&&Array.isArray(a)===Array.isArray(b)){
   assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),`${label}: fields at ${path}`);
   for(const key of Object.keys(b))visit(a[key],b[key],`${path}.${key}`);return;
  }
  assert.deepEqual(a,b,`${label}: ${path} differs (current ${JSON.stringify(a)}, reference ${JSON.stringify(b)}); Node ${process.version}, V8 ${process.versions.v8}, ${process.platform}/${process.arch}`);
 }
 visit(actual,expected,'output');
}
