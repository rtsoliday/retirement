import {runSimulation,estimateDecision} from './engine.js';
self.onmessage=e=>{try{const result=e.data.task==='decision'?estimateDecision(e.data.scenario):runSimulation(e.data.scenario);self.postMessage({type:'result',result});}catch(error){self.postMessage({type:'error',message:String(error.message||error)});}};
