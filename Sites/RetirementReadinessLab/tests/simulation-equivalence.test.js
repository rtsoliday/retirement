import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {JavaRandom,runOne,runSimulation} from '../dist/engine.js';
import * as reference from './reference/pre-optimization/engine.js';
import {verifyReference,assertReference} from './reference-checks.js';

// Independent pre-optimization implementation of java.util.Random. Pin every
// draw, including carry/wraparound, negative/large seeds and cached Gaussians.
class BigIntRandom{
  constructor(seed){this.seed=(BigInt(seed)^0x5deece66dn)&((1n<<48n)-1n);this.gaussian=null;}
  next(bits){this.seed=(this.seed*0x5deece66dn+0xbn)&((1n<<48n)-1n);return Number(this.seed>>BigInt(48-bits));}
  nextDouble(){return (this.next(26)*134217728+this.next(27))/9007199254740992;}
  nextGaussian(){if(this.gaussian!==null){const x=this.gaussian;this.gaussian=null;return x;}let v1,v2,s;do{v1=2*this.nextDouble()-1;v2=2*this.nextDouble()-1;s=v1*v1+v2*v2;}while(s>=1||s===0);const m=Math.sqrt(-2*Math.log(s)/s);this.gaussian=v2*m;return v1*m;}
  normal(mean,std){return mean+this.nextGaussian()*std;}
}

test('48-bit random transitions exactly match the BigInt reference',()=>{
  const seeds=[0n,1n,-1n,20260766n,0xffffffn,0x1000000n,0xffffffffffffn,1n<<48n,1n<<100n,-(1n<<100n)];
  for(const seed of seeds){
    const actual=new JavaRandom(seed),expected=new BigIntRandom(seed);
    for(let i=0;i<100000;i++)assert.equal(actual.next(i%49),expected.next(i%49));
    assert.equal(actual.seed,expected.seed);
  }
});

test('mixed uniform, Gaussian and zero-volatility draws retain the exact stream',()=>{
  for(const seed of [20260766n,-7046029254386353131n,0xffffffffffffn]){
    const actual=new JavaRandom(seed),expected=new BigIntRandom(seed);
    for(let i=0;i<10000;i++){
      assert.equal(actual.nextDouble(),expected.nextDouble());
      assert.equal(actual.nextGaussian(),expected.nextGaussian());
      assert.equal(actual.normal(.133,i%2?.162:0),expected.normal(.133,i%2?.162:0));
    }
    assert.equal(actual.seed,expected.seed);assert.equal(actual.gaussian,expected.gaussian);
  }
});

verifyReference('pre-optimization');
const scenarios=JSON.parse(readFileSync(new URL('./fixtures/simulation-reference-inputs.json',import.meta.url)));
for(const [name,s] of scenarios)test(`Exact pre-optimization results and monthly tax traces: ${name}`,()=>{
  function capture(engine){
    const result=engine.runSimulation(structuredClone(s));delete result.generatedAtEpochMillis;
    const paths=[0,1,7].map(i=>engine.runOne(structuredClone(s),new engine.JavaRandom(BigInt(s.seed)+BigInt(i)*-7046029254386353131n),{
      captureMonthlyBalances:true,captureMonthlyDetails:true,captureTaxDetails:true,captureTodayDollars:true
    }));return {result,paths};
  }
  assertReference(capture({JavaRandom,runOne,runSimulation}),capture(reference),name);
});
