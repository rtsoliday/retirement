import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,readdirSync} from 'node:fs';
import worker from '../worker/index.js';

const dist=new URL('../dist/',import.meta.url);
const read=name=>readFileSync(new URL(name,dist),'utf8');
const publicPages=readdirSync(dist).filter(name=>name.endsWith('.html')&&name!=='admin.html');
const canonical=name=>'https://retirementforecast.us'+(name==='index.html'?'/':'/'+name.replace(/\.html$/,''));

test('public pages identify the same preferred non-www URLs from the HTML head',()=>{
  for(const name of publicPages){
    const html=read(name),head=html.match(/<head>([\s\S]*?)<\/head>/i)?.[1];
    assert.ok(head,name);
    const tags=[...head.matchAll(/<link\b[^>]*rel="canonical"[^>]*>/g)];
    assert.equal(tags.length,1,name);
    assert.match(tags[0][0],new RegExp('href="'+canonical(name).replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'"'));
    assert.doesNotMatch(head,/<meta\b[^>]*name="robots"[^>]*content="[^"]*noindex/i,name);
  }
  assert.match(read('admin.html'),/name="robots" content="noindex, nofollow"/);
});

test('sitemap contains every public canonical page exactly once and excludes private routes',()=>{
  const xml=read('sitemap.xml');
  assert.match(xml,/^<\?xml version="1.0" encoding="UTF-8"\?>/);
  assert.match(xml,/<urlset xmlns="http:\/\/www.sitemaps.org\/schemas\/sitemap\/0.9">/);
  const urls=[...xml.matchAll(/<loc>(.*?)<\/loc>/g)].map(match=>match[1]);
  assert.deepEqual(urls.toSorted(),publicPages.map(canonical).toSorted());
  assert.equal(new Set(urls).size,urls.length);
  for(const value of urls){const url=new URL(value);assert.equal(url.origin,'https://retirementforecast.us');assert.doesNotMatch(url.pathname,/\.html$/);assert.equal(url.search,'');assert.equal(url.hash,'');}
  assert.doesNotMatch(xml,/admin|\/api\/|signin|signout/);
});

test('robots permits public crawling and advertises the working preferred sitemap',()=>{
  const robots=read('robots.txt');
  assert.match(robots,/^User-agent: \*$/m);assert.match(robots,/^Allow: \/$/m);
  assert.match(robots,/^Sitemap: https:\/\/retirementforecast.us\/sitemap.xml$/m);
  assert.doesNotMatch(robots,/^Disallow: \/\s*$/m);
  assert.match(robots,/^Disallow: \/admin$/m);assert.match(robots,/^Disallow: \/api\/$/m);
});

test('anonymous sitemap and robots requests on all site origins reach the static files',async()=>{
  for(const origin of ['https://retirementforecast.us','https://www.retirementforecast.us','https://retirement-readiness-lab-web.rtsoliday123.chatgpt.site']){
    for(const name of ['sitemap.xml','robots.txt']){
      let assetURL;
      const env={ASSETS:{async fetch(request){assetURL=request.url;const file=new URL(request.url).pathname.slice(1);return new Response(read(file),{headers:{'Content-Type':file.endsWith('.xml')?'application/xml':'text/plain'}});}}};
      const response=await worker.fetch(new Request(origin+'/'+name),env);
      assert.equal(response.status,200);assert.equal(assetURL,origin+'/'+name);
      assert.equal(response.headers.get('X-Robots-Tag'),null);assert.equal(await response.text(),read(name));
    }
  }
});
