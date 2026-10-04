import {runSimulation,searchDecision,candidateReadiness} from './engine.js';
import {searchReadinessFrontier,searchSavingsFrontier} from './target-frontier.js';

// Spread target-search candidates across nested workers when the browser allows
// them. Any worker that fails to load hands its work back to this worker, so the
// search still finishes with the same answer.
function candidatePool(scenario){
  const inline=task=>Promise.resolve().then(()=>candidateReadiness(scenario,task));
  const size=Math.min(8,(self.navigator?.hardwareConcurrency||1)-1);
  if(size<2||typeof Worker!=='function')return {size:1,evaluate:inline,close(){}};
  const idle=[],queue=[],jobs=new Map(),workers=[];let broken=false;
  const dispatch=()=>{
    while(queue.length&&(broken||idle.length)){
      const job=queue.shift();
      if(broken){inline(job.task).then(job.resolve,job.reject);continue;}
      // First in, first out: the earliest candidates go to the workers that loaded first.
      const worker=idle.shift();jobs.set(worker,job);worker.postMessage({task:'decision-candidate',scenario,candidate:job.task});
    }
  };
  for(let i=0;i<size;i++){
    let worker;
    try{worker=new Worker(new URL('./worker.js',import.meta.url),{type:'module'});}catch{break;}
    worker.onmessage=e=>{
      const job=jobs.get(worker);jobs.delete(worker);idle.push(worker);
      if(e.data.type==='result')job.resolve(e.data.result);else job.reject(new Error(e.data.message));
      dispatch();
    };
    worker.onerror=e=>{
      e.preventDefault?.();broken=true;worker.terminate();
      const job=jobs.get(worker);jobs.delete(worker);if(job)queue.unshift(job);
      dispatch();
    };
    workers.push(worker);idle.push(worker);
  }
  if(!workers.length)return {size:1,evaluate:inline,close(){}};
  return {size:workers.length,evaluate:task=>new Promise((resolve,reject)=>{queue.push({task,resolve,reject});dispatch();}),close(){for(const worker of workers)worker.terminate();}};
}

self.onmessage=async e=>{
  const {task,scenario}=e.data,progress=update=>self.postMessage({type:'progress',...update});
  try{
    let result;
    if(task==='decision-candidate')result=candidateReadiness(scenario,e.data.candidate);
    else if(task==='claim-decision'||task==='savings-decision'){
      const pool=candidatePool(scenario);
      try{result={frontier:task==='savings-decision'?await searchSavingsFrontier(scenario,{evaluate:pool.evaluate,onProgress:progress}):await searchReadinessFrontier(scenario,{kind:'claiming',evaluate:pool.evaluate,onProgress:progress})};}
      finally{pool.close();}
    }
    else if(task==='decision'){
      const pool=candidatePool(scenario);
      try{
        result=await searchDecision(scenario,{evaluate:pool.evaluate,concurrency:pool.size,onProgress:update=>progress({...update,frontierExpected:true})});
        result.frontier=await searchReadinessFrontier(scenario,{evaluate:pool.evaluate,simulationCount:result.simulationCount,targetReadiness:result.targetReadiness,initialSpending:result.safeAnnualSpending??scenario.spending.annualBaseSpending,onProgress:progress});
      }
      finally{pool.close();}
    }
    else result=runSimulation(scenario,fraction=>progress({fraction}));
    self.postMessage({type:'result',result});
  }catch(error){self.postMessage({type:'error',message:String(error.message||error)});}
};
