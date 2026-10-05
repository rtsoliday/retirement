import test from 'node:test';
import assert from 'node:assert/strict';
import {PersonAccounts} from '../dist/person-accounts.js';
import {simulationScenarios} from './fixtures/simulation-scenarios.js';

for(const[name,original]of simulationScenarios())test(`withdrawal probes preserve taxes, penalties and ledgers: ${name}`,()=>{
  for(const order of ['Standard','TaxableFirst','RothLast'])for(const month of [0,1,59,60,120,240])for(const dead of [false,true]){
    const s=structuredClone(original);s.withdrawalStrategy.withdrawalOrder=order;
    const pools=new PersonAccounts(s,{...s.accounts});
    pools.configure(month,2026+Math.floor(month/12),[0,100],[!dead,true],()=>6000);
    for(const cashFirst of [false,true])for(const conversionTax of [false,true])for(const gross of [0,0.01,1000,10000,100000,1000000,1e9]){
      const state=JSON.stringify(pools),options={cashFirst,conversionTax},full=pools.quote(gross,options);
      assert.deepEqual(pools.quote(gross,{...options,capture:false}),{taxableDraw:full.taxableDraw,rothTaxableEarnings:full.rothTaxableEarnings,penalties:full.penalties},`${order}, month ${month}, gross ${gross}`);
      assert.equal(JSON.stringify(pools),state,'Search probes and quotes must not mutate balances or Roth histories');
    }
  }
});
