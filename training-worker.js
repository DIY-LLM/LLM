// training-worker.js
// Runs in a background thread for non-blocking UI training with IndexedDB persistence
// ====

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js');

// ====================================================================
// CONFIG and State
// ====================================================================

const CONFIG = {
    // These should be kept in sync with the main thread's CONFIG
    SEQUENCE_LENGTH: 10,
    UNK_TOKEN_ID: 0,
    MAX_CHECKPOINT_SIZE_MB: 50,
    DB_NAME: 'LLMTrainingDB',
    STORE_NAME: 'Checkpoints',
    // training config will be passed on START
};

let workerModel = null;
let isTraining = false;
let taskId = 'training-task'; 

// ====================================================================
// IndexedDB Persistence Functions (Section 1.2 & 3.3)
// ====================================================================

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(CONFIG.DB_NAME, 1);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            // Create the object store with 'taskId' as the key
            if (!db.objectStoreNames.contains(CONFIG.STORE_NAME)) {
                 db.createObjectStore(CONFIG.STORE_NAME, { keyPath: 'taskId' });
            }
        };

        request.onsuccess = (event) => resolve(event.target.result);
        request.onerror = (event) => reject(event.target.error);
    });
}

// Checkpointing Responsibility: Web Worker writes directly to IndexedDB
async function saveCheckpoint(taskId, epoch, loss, weightsBuffer) {
    if (weightsBuffer.byteLength > CONFIG.MAX_CHECKPOINT_SIZE_MB * 1024 * 1024) {
        throw new Error(`Checkpoint size exceeds limit of ${CONFIG.MAX_CHECKPOINT_SIZE_MB} MB.`);
    }

    const db = await openDB();
    // Use an explicit transaction
    const transaction = db.transaction([CONFIG.STORE_NAME], 'readwrite');
    const store = transaction.objectStore(CONFIG.STORE_NAME);

    const checkpoint = {
        taskId: taskId,
        timestamp: Date.now(),
        stepNumber: epoch, // The precise iteration/epoch/step to resume from.
        currentLoss: loss,
        modelWeights: new Blob([weightsBuffer]) // Store weights as a Blob
    };
    
    // Handle Quota Error (Section 3.3)
    try {
        await new Promise((resolve, reject) => {
            const request = store.put(checkpoint);
            request.onsuccess = () => resolve();
            request.onerror = (e) => {
                if (e.target.error.name === 'QuotaExceededError') {
                     reject(e.target.error); 
                } else {
                    reject(e.target.error);
                }
            };
        });
    } catch (e) {
        if (e.name === 'QuotaExceededError') {
             isTraining = false;
             postMessage({ command: 'QUOTA_ERROR' });
             throw e; // Stop execution
        }
        throw e;
    }
}

async function getLatestCheckpoint(taskId) {
    const db = await openDB();
    const transaction = db.transaction([CONFIG.STORE_NAME], 'readonly');
    const store = transaction.objectStore(CONFIG.STORE_NAME);

    return new Promise((resolve) => {
        const request = store.get(taskId);
        request.onsuccess = (event) => {
            const checkpoint = event.target.result;
            if (checkpoint && checkpoint.modelWeights instanceof Blob) {
                // Convert Blob back to ArrayBuffer
                const reader = new FileReader();
                reader.onload = () => {
                     checkpoint.modelWeights = reader.result;
                     resolve(checkpoint);
                };
                reader.readAsArrayBuffer(checkpoint.modelWeights);
            } else {
                 resolve(checkpoint);
            }
        };
        request.onerror = () => resolve(null);
    });
}

async function deleteCheckpoint(taskId) {
    const db = await openDB();
    const transaction = db.transaction([CONFIG.STORE_NAME], 'readwrite');
    const store = transaction.objectStore(CONFIG.STORE_NAME);
    await new Promise(resolve => {
        const request = store.delete(taskId);
        request.onsuccess = resolve;
        request.onerror = resolve;
    });
}


// ====================================================================
// Transformer Model Definition
// ====================================================================

function createModel(config) {
    const { vocabSize, sequenceLength, embeddingDim, ffnDim, numHeads, learningRate } = config;
    
    // 1. Input Layer
    const input = tf.input({ shape: [sequenceLength], dtype: 'int32' });
    
    // 2. Embedding Layer
    let embedding = tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embeddingDim,
        inputLength: sequenceLength
    }).apply(input);

    // 3. Simplified Transformer Block (Self-Attention)
    const attentionOutput = tf.layers.multiHeadAttention({
        numHeads: numHeads,
        keyDim: embeddingDim / numHeads 
    }).apply([embedding, embedding, embedding]); 

    // Residual Connection and Layer Normalization
    let norm1 = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(tf.add(embedding, attentionOutput));
    
    // Feed Forward Network
    const ffn = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: ffnDim, activation: 'relu' })
    }).apply(norm1);
    
    const ffnOutput = tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: embeddingDim })
    }).apply(ffn);
    
    let norm2 = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(tf.add(norm1, ffnOutput));

    // 4. Output Layer: Use the last token's representation for next word prediction (typical LLM approach)
    const finalVector = tf.layers.flatten({}).apply(norm2);

    // Final Dense layer to map to vocabulary size (logits)
    const output = tf.layers.dense({ 
        units: vocabSize,
        activation: 'softmax' 
    }).apply(finalVector);
    
    const model = tf.model({ inputs: input, outputs: output });

    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });
    
    return model;
}

// ====================================================================
// Training Function (Section 3.2)
// ====================================================================

async function trainWorkerModel(data) {
    const config = data.config;
    taskId = config.taskId; // Update global taskId
    
    // Convert ArrayBuffers back to Tensors
    const flatInputs = new Float32Array(data.data.inputsBuffer);
    const flatTargets = new Float32Array(data.data.targetsBuffer);
    
    const xTrain = tf.tensor2d(flatInputs, data.data.inputShape, 'int32');
    const yTrain = tf.tensor1d(flatTargets, 'int32');
    
    let currentEpoch = 0;
    
    // 1. Load or Create Model
    if (!workerModel) {
        workerModel = createModel(config);
    }

    // 2. Resumption Logic (Section 3.1)
    if (data.command === 'RESUME') {
        const checkpoint = await getLatestCheckpoint(taskId);
        if (checkpoint) {
            currentEpoch = checkpoint.stepNumber;
            
            // The file 2 snippet provided the weight serialization logic:
            // weightData = weights.map(w => ({ name: w.name, data: w.dataSync(), dtype: w.dtype, shape: w.shape }))
            const weightData = JSON.parse(new TextDecoder().decode(new Uint8Array(checkpoint.modelWeights)));
            
            const weights = weightData.map(w => ({
                name: w.name,
                data: tf.tensor(w.data, w.shape, w.dtype)
            }));
            workerModel.setWeights(weights);
            
            postMessage({ command: 'PROGRESS', epoch: currentEpoch, loss: checkpoint.currentLoss, progress: 'Restored Checkpoint' });
        }
    }
    
    isTraining = true;
    
    try {
        await workerModel.fit(xTrain, yTrain, {
            epochs: config.epochs,
            initialEpoch: currentEpoch, 
            batchSize: config.batchSize,
            callbacks: [
                {
                    // Runtime Checkpointing (Saving Progress)
                    onEpochEnd: async (epoch, logs) => {
                        if (!isTraining) {
                            workerModel.stopTraining = true;
                            return;
                        }
                        
                        const currentEpoch = epoch + 1;
                        
                        // Send minimal progress update to main thread
                        postMessage({ command: 'PROGRESS', epoch: currentEpoch, loss: logs.loss, progress: 'Epoch Complete' });
                        
                        // Save checkpoint at a regular, non-blocking interval (every 5 epochs)
                        if (currentEpoch % 5 === 0 || currentEpoch === config.epochs) {
                            const weights = workerModel.getWeights();
                            const weightData = weights.map(w => ({
                                name: w.name,
                                data: w.dataSync(),
                                dtype: w.dtype,
                                shape: w.shape
                            }));
                            
                            // Convert to ArrayBuffer for efficient storage
                            const jsonWeights = new TextEncoder().encode(JSON.stringify(weightData));

                            // Save operation must be transactional (handled in saveCheckpoint)
                            await saveCheckpoint(taskId, currentEpoch, logs.loss, jsonWeights.buffer);
                        }
                    }
                }
            ]
        });

        // Only fire COMPLETE if it wasn't stopped manually
        if (isTraining) { 
            postMessage({ command: 'COMPLETE' });
        }
    } catch (e) {
        if (e.message !== 'model.stopTraining is true.') {
            // Worker Error Handling (Section 3.3)
            postMessage({ command: 'ERROR', message: e.message });
            console.error(e);
        }
    } finally {
        isTraining = false;
        // Clean up tensors
        tf.disposeVariables();
    }
}

// ====================================================================
// Message Handler
// ====================================================================

self.onmessage = async (event) => {
    const data = event.data;
    switch (data.command) {
        case 'START':
        case 'RESUME':
            await trainWorkerModel(data);
            break;
        case 'STOP':
            isTraining = false;
            if (workerModel) {
                 workerModel.stopTraining = true;
            }
            break;
        case 'CHECK_CHECKPOINT': // Used in Startup Flow (Section 3.1)
            const checkpoint = await getLatestCheckpoint(data.taskId);
            if (checkpoint) {
                postMessage({ command: 'CHECKPOINT_FOUND', taskId: checkpoint.taskId, stepNumber: checkpoint.stepNumber });
            } else {
                postMessage({ command: 'NO_CHECKPOINT' });
            }
            break;
        case 'DELETE_CHECKPOINT':
            await deleteCheckpoint(data.taskId);
            break;
    }
};
