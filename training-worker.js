// =========================================================================
// training-worker.js
// Runs in a background thread for non-blocking UI training with IndexedDB persistence
// =========================================================================

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js');

// Constants (replicated from index.html)
const CONFIG = {
    // These should be kept in sync with the main thread's CONFIG
    SEQUENCE_LENGTH: 10,
    UNK_TOKEN_ID: 0,
    MAX_CHECKPOINT_SIZE_MB: 50,
    DB_NAME: 'LLMTrainingDB',
    STORE_NAME: 'Checkpoints',
    train: {
        epochs: 20,
        batchSize: 16,
        learningRate: 0.0005
    }
    // ... other model config ...
};

let workerModel = null;
let currentTrainingJob = null;
let checkpointMetadata = null;
let tokenizerState = { vocab: new Map(), idToToken: new Map() };
let isTraining = false;
let tfjsInitialized = false;

// =========================================================================
// IndexedDB Persistence
// =========================================================================

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(CONFIG.DB_NAME, 1);
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            db.createObjectStore(CONFIG.STORE_NAME, { keyPath: 'taskId' });
        };
        request.onsuccess = (event) => resolve(event.target.result);
        request.onerror = (event) => reject(event.target.error);
    });
}

async function saveCheckpoint(taskId, epoch, loss, weights) {
    if ((weights.byteLength / 1024 / 1024) > CONFIG.MAX_CHECKPOINT_SIZE_MB) {
        postMessage({ command: 'QUOTA_ERROR', message: `Model size exceeds ${CONFIG.MAX_CHECKPOINT_SIZE_MB}MB limit.` });
        return;
    }

    try {
        const db = await openDB();
        const transaction = db.transaction([CONFIG.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(CONFIG.STORE_NAME);

        const checkpoint = {
            taskId: taskId,
            timestamp: Date.now(),
            epoch: epoch,
            loss: loss,
            modelWeights: weights, // ArrayBuffer or Blob
            config: CONFIG.train // Save training config
        };

        await new Promise((resolve, reject) => {
            const request = store.put(checkpoint);
            request.onsuccess = resolve;
            request.onerror = (event) => {
                // Check specifically for QuotaExceededError
                if (event.target.error.name === 'QuotaExceededError') {
                    postMessage({ command: 'QUOTA_ERROR' });
                }
                reject(event.target.error);
            };
        });

        console.log(`Checkpoint saved for Task ${taskId} at Epoch ${epoch}.`);
        checkpointMetadata = { taskId, epoch, loss }; // Update metadata
    } catch (e) {
        if (e.name === 'QuotaExceededError') {
             postMessage({ command: 'QUOTA_ERROR' });
        } else {
             console.error("Error saving checkpoint:", e);
        }
    }
}

async function loadCheckpoint(taskId) {
    try {
        const db = await openDB();
        const transaction = db.transaction([CONFIG.STORE_NAME], 'readonly');
        const store = transaction.objectStore(CONFIG.STORE_NAME);
        
        return await new Promise((resolve, reject) => {
            const request = store.get(taskId);
            request.onsuccess = (event) => resolve(event.target.result);
            request.onerror = (event) => reject(event.target.error);
        });
    } catch (e) {
        console.error("Error loading checkpoint:", e);
        return null;
    }
}

async function checkExistingCheckpoint() {
    try {
        const db = await openDB();
        const transaction = db.transaction([CONFIG.STORE_NAME], 'readonly');
        const store = transaction.objectStore(CONFIG.STORE_NAME);

        // Simple check for the first available checkpoint
        const request = store.openCursor();
        await new Promise((resolve) => {
            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    postMessage({ 
                        command: 'CHECKPOINT_FOUND', 
                        taskId: cursor.value.taskId,
                        epoch: cursor.value.epoch,
                        loss: cursor.value.loss
                    });
                }
                resolve(); // resolve even if no cursor
            };
            request.onerror = () => resolve();
        });

    } catch (e) {
        console.error("Error checking for checkpoint:", e);
    }
}

async function deleteCheckpoint(taskId) {
    try {
        const db = await openDB();
        const transaction = db.transaction([CONFIG.STORE_NAME], 'readwrite');
        const store = transaction.objectStore(CONFIG.STORE_NAME);
        await new Promise((resolve, reject) => {
            const request = store.delete(taskId);
            request.onsuccess = resolve;
            request.onerror = reject;
        });
        checkpointMetadata = null;
        console.log(`Checkpoint ${taskId} deleted.`);
    } catch (e) {
        console.error("Error deleting checkpoint:", e);
    }
}


// =========================================================================
// Model & Training Logic (Simplified for Worker)
// =========================================================================

function createWorkerModel(vocabSize, embeddingDim, sequenceLength) {
    // Simplified Attention/FFN block (same as main thread for consistency)
    const attentionBlock = (input, heads, ffnDim) => {
        // ... (Model layers) ...
        const headSize = embeddingDim / heads;
                
        let att = tf.layers.multiHeadAttention({ 
            numHeads: heads, 
            keyDim: headSize 
        }).apply([input, input, input]);
        
        att = tf.layers.dropout({ rate: 0.2 }).apply(att);
        att = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(tf.layers.add().apply([input, att]));
        
        let ffn = tf.layers.dense({ units: ffnDim, activation: 'relu' }).apply(att);
        ffn = tf.layers.dense({ units: embeddingDim }).apply(ffn);
        ffn = tf.layers.dropout({ rate: 0.2 }).apply(ffn);
        return tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(tf.layers.add().apply([att, ffn]));
    };

    const input = tf.input({ shape: [sequenceLength] });
    let embedding = tf.layers.embedding({ 
        inputDim: vocabSize, 
        outputDim: embeddingDim, 
        embeddingsRegularizer: tf.regularizers.l2({ l2: 1e-5 })
    }).apply(input);

    const posEmbedding = tf.layers.embedding({
        inputDim: sequenceLength,
        outputDim: embeddingDim,
        trainable: true 
    }).apply(tf.range(0, sequenceLength).expandDims(0).tile([tf.shape(embedding).arraySync()[0], 1]));
    embedding = tf.layers.add().apply([embedding, posEmbedding]);

    let output = attentionBlock(embedding, 4, 256);
    output = attentionBlock(output, 4, 256);
    output = attentionBlock(output, 4, 256);
    output = attentionBlock(output, 4, 256);

    output = tf.layers.flatten().apply(output);
    output = tf.layers.dense({ units: vocabSize, activation: 'softmax' }).apply(output);

    workerModel = tf.model({ inputs: input, outputs: output });

    workerModel.compile({
        optimizer: tf.train.adam(CONFIG.train.learningRate),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    console.log("Worker MiniTransformer Model Created.");
}

async function trainWorkerModel(data) {
    if (isTraining) {
        postMessage({ command: 'ERROR', message: 'Training already in progress.' });
        return;
    }

    isTraining = true;

    // Restore tokenizer state from main thread
    tokenizerState.vocab = new Map(data.vocab.vocab);
    tokenizerState.idToToken = new Map(data.vocab.idToToken);
    const vocabSize = tokenizerState.vocab.size;
    const seqLen = CONFIG.SEQUENCE_LENGTH;
    const { epochs, batchSize } = CONFIG.train;

    // Simple tokenization of the corpus
    const tokenize = (text) => {
        const tokens = text.toLowerCase().match(/\b\w+\b/g) || [];
        return tokens.map(token => tokenizerState.vocab.get(token) || CONFIG.UNK_TOKEN_ID);
    };
    const tokenIds = tokenize(data.corpus);
    
    // Prepare sequential data
    const xs = [];
    const ys = [];
    for (let i = 0; i < tokenIds.length - seqLen; i++) {
        xs.push(tokenIds.slice(i, i + seqLen));
        ys.push(tokenIds[i + seqLen]);
    }
    const numSamples = xs.length;

    if (numSamples === 0) {
        postMessage({ command: 'ERROR', message: 'Not enough data to form sequences.' });
        isTraining = false;
        return;
    }

    // Convert to Tensors
    const xTrain = tf.tensor2d(xs, [numSamples, seqLen]);
    const yTrain = tf.oneHot(tf.tensor1d(ys, 'int32'), vocabSize); 
    let currentEpoch = 0;
    const taskId = data.resume ? data.checkpointId : `task-${Date.now()}`;
    let saveTimer = null;

    if (!workerModel) {
        createWorkerModel(vocabSize, 128, seqLen); // Using default embeddingDim
    }

    // Handle Checkpoint Resumption
    if (data.resume && data.checkpointId) {
        const checkpoint = await loadCheckpoint(data.checkpointId);
        if (checkpoint && checkpoint.modelWeights) {
            // Restore weights from ArrayBuffer
            const weightMap = tf.io.decodeWeights(new Uint8Array(checkpoint.modelWeights));
            workerModel.setWeights(weightMap);
            currentEpoch = checkpoint.epoch;
            console.log(`Resumed from checkpoint. Starting at Epoch ${currentEpoch + 1}.`);
        }
    }


    const onBatchEnd = (batch, logs) => {
        if (!isTraining) {
            workerModel.stopTraining = true;
        }

        const totalSteps = Math.ceil(numSamples / batchSize);
        postMessage({
            command: 'PROGRESS',
            epoch: currentEpoch + 1,
            totalEpochs: epochs,
            step: batch + 1,
            totalSteps: totalSteps,
            loss: logs.loss,
            memory: tf.memory().numBytes
        });
    };

    const onEpochEnd = async (epoch, logs) => {
        currentEpoch = epoch;
        
        // Non-blocking checkpoint save every 5 minutes (or on demand)
        // Here we just save at the end of every epoch for simplicity in this demo.
        await tf.nextFrame(); // Yield control to the main thread briefly

        // Get weights and save them to IndexedDB
        const weights = await workerModel.save(tf.io.withSaveHandler(async artifacts => {
            // We only need the ArrayBuffer for the binary weights file
            return artifacts.weightData; 
        }));

        // The save method above is complex. For a worker, it's easier to:
        const weightData = tf.io.encodeWeights(workerModel.getWeights().map(w => ({
            name: w.name,
            data: w.dataSync(),
            dtype: w.dtype,
            shape: w.shape
        })));

        await saveCheckpoint(taskId, currentEpoch + 1, logs.loss, weightData.buffer);

        console.log(`Epoch ${currentEpoch + 1} complete. Loss: ${logs.loss.toFixed(4)}. Checkpoint saved.`);
    };

    try {
        await workerModel.fit(xTrain, yTrain, {
            epochs: epochs,
            initialEpoch: currentEpoch, // Start from where we left off
            batchSize: batchSize,
            callbacks: { onBatchEnd, onEpochEnd }
        });

        if (isTraining) {
            // Only fire COMPLETE if it wasn't stopped manually
            postMessage({ command: 'COMPLETE' });
        }
    } catch (e) {
        if (e.message === 'model.stopTraining is true.') {
            // Manually stopped, not an error
        } else {
            postMessage({ command: 'ERROR', message: e.message });
            console.error(e);
        }
    } finally {
        isTraining = false;
        xTrain.dispose();
        yTrain.dispose();
    }
}

// =========================================================================
// Message Handler
// =========================================================================

self.onmessage = async (event) => {
    const data = event.data;

    switch (data.command) {
        case 'START':
            trainWorkerModel(data);
            break;
        case 'STOP':
            isTraining = false; // Stop the fit loop
            break;
        case 'CHECK_CHECKPOINT':
            checkExistingCheckpoint();
            break;
        case 'DELETE_CHECKPOINT':
            deleteCheckpoint(data.taskId);
            break;
        case 'GET_WEIGHTS':
            // ... (Logic to return model weights for saving/export) ...
            break;
        // In a real app, you would have an INIT command to send the full config
    }
};

// Notify main thread that the worker script is loaded and ready
postMessage({ command: 'READY' });
