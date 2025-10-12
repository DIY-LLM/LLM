/* training-worker.js
Dedicated Web Worker:
- Accepts commands: start-new, resume, pause, save-now
- Runs a CPU-bound training loop (toy model) but structured to be replaceable with real training
- Periodically (every 5s) posts progress messages
- Checkpoints state to IndexedDB transactionally; handles QuotaExceededError and posts checkpoint-error
- Checkpoint structure: { taskId, timestamp, stepNumber, taskConfig, modelWeights: Blob }
*/

const DB_NAME = 'llm_worker_checkpoints_v1';
const STORE_NAME = 'checkpoints';

// Worker state
let state = {
  running: false,
  paused: false,
  taskId: null,
  stepNumber: 0,
  targetEpochs: 50,
  saveInterval: 300000, // default 5m
  lastSaveTs: 0,
  msPerStep: 50, // simulated work duration per step
  modelWeights: null // Float32Array
};

let periodicProgressTimer = null;
let checkpointTimer = null;
let lastProgressPost = 0;

// utility: post message safely
function send(msg){
  postMessage(msg);
}

// IndexedDB helpers (worker context supports indexedDB)
function openDb(){
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)){
        db.createObjectStore(STORE_NAME, {keyPath: 'taskId'});
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('IDB open failed'));
  });
}

async function saveCheckpointTransactional(){
  // Serialize modelWeights to Blob (ArrayBuffer)
  try {
    const db = await openDb();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);

    const modelArray = state.modelWeights || new Float32Array([Math.random()]); // Fallback
    const ab = modelArray.buffer ? modelArray.buffer : (new Float32Array(modelArray)).buffer;
    const blob = new Blob([ab], {type:'application/octet-stream'});

    const record = {
      taskId: state.taskId || ('task-'+Date.now()),
      timestamp: Date.now(),
      stepNumber: state.stepNumber,
      taskConfig: { targetEpochs: state.targetEpochs, msPerStep: state.msPerStep },
      modelWeights: blob
    };

    return new Promise((resolve, reject) => {
      const req = store.put(record);
      req.onsuccess = () => {
        resolve(record);
      };
      req.onerror = (e) => {
        reject(req.error || new Error('IDB put failed'));
      };
      // safeguard: abort on tx error
      tx.onabort = () => reject(new Error('IDB transaction aborted'));
    });
  } catch(err){
    throw err;
  }
}

async function loadLatestCheckpoint(taskId){
  // Load specified taskId or newest
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    if (taskId){
      const r = store.get(taskId);
      r.onsuccess = ()=> resolve(r.result || null);
      r.onerror = ()=> reject(r.error);
    } else {
      const cursor = store.openCursor();
      let best = null;
      cursor.onsuccess = (e)=>{
        const c = e.target.result;
        if (!c) { resolve(best); return; }
        const rec = c.value;
        if (!best || (rec.timestamp || 0) > (best.timestamp || 0)) best = rec;
        c.continue();
      };
      cursor.onerror = ()=>reject(cursor.error);
    }
  });
}

function setModelRandom(size=256){
  // small model: Float32Array
  state.modelWeights = new Float32Array(size);
  for (let i=0;i<size;i++) state.modelWeights[i] = (Math.random()-0.5)*0.1;
}

// Simulated training step (CPU-bound but small)
function trainingStep(){
  // simple gradient-like update of the Float32Array weights
  const w = state.modelWeights;
  let loss = 0;
  for (let i=0;i<w.length;i++){
    // pretend gradient is sign of weight + small noise
    const grad = Math.tanh(w[i]) * 0.01 + (Math.random()-0.5)*0.001;
    w[i] -= grad * 0.5; // step
    loss += Math.abs(w[i]);
  }
  // return pseudo loss
  return loss / w.length;
}

async function runLoop(){
  state.running = true;
  state.paused = false;
  send({type:'log', text:'Training loop started.'});
  // start periodic progress (every 5s)
  lastProgressPost = Date.now();
  if (periodicProgressTimer) clearInterval(periodicProgressTimer);
  periodicProgressTimer = setInterval(()=>{
    postProgress();
  }, 5000);

  // checkpoint timer (in-case saveInterval is not tied to iterations)
  if (checkpointTimer) clearInterval(checkpointTimer);
  checkpointTimer = setInterval(()=>{
    try {
      checkpointNow().catch(e=>{
        // post error; main will handle QuotaExceededError case
        send({type:'checkpoint-error', error:serializeError(e)});
      });
    } catch(e){
      send({type:'checkpoint-error', error:serializeError(e)});
    }
  }, Math.max(1000, state.saveInterval || 300000));

  // Main training loop
  while(state.running){
    if (state.paused){
      await sleep(200);
      continue;
    }
    // perform one "step" (simulated workload)
    const start = performance.now();
    const loss = trainingStep();
    const end = performance.now();
    state.stepNumber += 1;

    // occasionally send progress if >5s passed
    if (Date.now() - lastProgressPost >= 5000){
      postProgress(loss);
      lastProgressPost = Date.now();
    }

    // checkpointing by step-count threshold (e.g., every 100 steps)
    if (state.stepNumber % 100 === 0){
      try {
        const meta = await checkpointNow();
        send({type:'checkpoint-saved', meta});
      } catch(err){
        // handle quota exceeded specially:
        send({type:'checkpoint-error', error:serializeError(err)});
        if (err && err.name === 'QuotaExceededError'){
          // Pause worker and let main thread propose download
          state.paused = true;
          send({type:'log', text:'Paused due to QuotaExceededError.'});
        }
      }
    }

    // finish condition
    if (state.stepNumber >= state.targetEpochs){
      send({type:'log', text:'Target epochs reached. Training complete.'});
      postProgress();
      await checkpointNow().catch(e=>send({type:'checkpoint-error', error:serializeError(e)}));
      state.running = false;
      break;
    }

    // throttle to simulate real CPU time (msPerStep)
    const elapsed = (end - start);
    const toWait = Math.max(0, state.msPerStep - elapsed);
    if (toWait > 0) await sleep(toWait);
  }

  // cleanup timers
  if (periodicProgressTimer) { clearInterval(periodicProgressTimer); periodicProgressTimer=null; }
  if (checkpointTimer) { clearInterval(checkpointTimer); checkpointTimer=null; }

  send({type:'log', text:'Training loop exited.'});
}

function postProgress(loss){
  const pct = Math.min(100, Math.round((state.stepNumber / (state.targetEpochs || 1)) * 100));
  send({
    type:'progress',
    percentage: pct,
    stepNumber: state.stepNumber,
    epochs: state.targetEpochs,
    status: state.paused ? 'paused' : (state.running ? 'running' : 'idle'),
    loss: loss || null
  });
}

async function checkpointNow(){
  // Save to IDB
  try {
    const meta = await saveCheckpointTransactional();
    state.lastSaveTs = Date.now();
    return meta;
  } catch(err){
    // If QuotaExceededError bubble up error object with name property
    throw err;
  }
}

// helper
function sleep(ms){ return new Promise(r=>setTimeout(r, ms)); }

function serializeError(e){
  if (!e) return null;
  return {name:e.name, message:e.message, stack: e.stack};
}

// message handler
onmessage = async (ev) => {
  const msg = ev.data;
  if (!msg || !msg.type) return;
  try {
    switch(msg.type){
      case 'start-new':
        {
          const p = msg.payload || {};
          state.taskId = p.taskId || ('task-'+Date.now());
          const cfg = p.config || {};
          state.targetEpochs = Number(cfg.epochs) || 50;
          state.saveInterval = Number(cfg.saveInterval) || state.saveInterval;
          state.stepNumber = 0;
          setModelRandom(1024); // smallish model by default
          send({type:'log', text:'Starting new training task ' + state.taskId});
          postMessage({type:'ready'});
          // start loop
          runLoop().catch(e=>send({type:'log', text:'RunLoop failed: '+ (e.message||e)}));
        }
        break;
      case 'resume':
        {
          const taskId = msg.payload && msg.payload.taskId;
          // attempt to load checkpoint
          try {
            const rec = await loadLatestCheckpoint(taskId || null);
            if (!rec) {
              send({type:'log', text:'No checkpoint found to resume.'});
              // optionally start new idle state
              break;
            }
            // read blob into ArrayBuffer then Float32Array
            const blob = rec.modelWeights;
            const ab = await blob.arrayBuffer();
            const fa = new Float32Array(ab);
            state.modelWeights = fa;
            state.stepNumber = rec.stepNumber || 0;
            state.targetEpochs = (rec.taskConfig && rec.taskConfig.targetEpochs) || state.targetEpochs;
            state.saveInterval = (rec.taskConfig && rec.taskConfig.saveInterval) || state.saveInterval;
            state.taskId = rec.taskId || state.taskId;
            send({type:'resumed', step: state.stepNumber});
            postProgress();
            // start loop
            runLoop().catch(e=>send({type:'log', text:'RunLoop failed after resume: '+e.message}));
          } catch(err){
            send({type:'log', text:'Resume failed: ' + (err && err.message)});
            send({type:'checkpoint-error', error:serializeError(err)});
          }
        }
        break;
      case 'pause':
        state.paused = true;
        send({type:'log', text:'Worker paused by main thread.'});
        postProgress();
        break;
      case 'unpause':
        state.paused = false;
        send({type:'log', text:'Worker unpaused by main thread.'});
        postProgress();
        break;
      case 'save-now':
        try {
          const meta = await checkpointNow();
          send({type:'checkpoint-saved', meta});
        } catch(err){
          send({type:'checkpoint-error', error:serializeError(err)});
        }
        break;
      default:
        send({type:'log', text:'Unknown command: ' + msg.type});
    }
  } catch(err){
    send({type:'log', text:'Unhandled worker exception: ' + (err && err.message)});
    send({type:'checkpoint-error', error:serializeError(err)});
  }
};

// initial ready message
postMessage({type:'ready'});

