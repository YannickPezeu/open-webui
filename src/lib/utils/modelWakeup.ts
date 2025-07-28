// $lib/utils/modelWakeup.ts - FIXED VERSION
import { writable } from 'svelte/store'; // Import writable
import { toast } from 'svelte-sonner';
import { modelsLoaded } from '$lib/stores';
import { WEBUI_API_BASE_URL } from '$lib/constants';
import { get } from 'svelte/store';

// Type definitions
interface Model {
  id: string;
  info?: {
    base_model_id?: string;
  };
}

interface I18n {
  t: (key: string, params?: Record<string, any>) => string;
}

interface WakeUpResponse {
  status?: string;
  chat_model?: {
    status?: string;
    success?: boolean;
    needs_wakeup?: boolean;
    name?: string;
  };
  embedding_model?: {
    status?: string;
    success?: boolean;
    needs_wakeup?: boolean;
    name?: string;
  };
  reranker_model?: {
    status?: string;
    success?: boolean;
    needs_wakeup?: boolean;
    name?: string;
  };
}

// Set to track models currently being woken up
const wakingUpModels = new Set<string>();
// Set to track models that have been recently attempted (with timestamps)
const recentWakeupAttempts = new Map<string, number>();

const WAKEUP_COOLDOWN = 5000; // 5 seconds cooldown

// Helper function to resolve actual model ID
export const resolveActualModelId = (modelId: string, models: Model[]): string => {
  const model = models.find(m => m.id === modelId);
  if (!model) return modelId;

  // If it's a level 1 model (has base_model_id), return the base model
  if (model.info?.base_model_id) {
    return model.info.base_model_id;
  }

  // If it's a level 0 model, return the original ID
  return model.id;
};

// Helper function to set model loading state
export const setModelLoaded = (modelId: string, loaded: boolean): void => {
  modelsLoaded.update(current => ({
    ...current,
    [modelId]: loaded
  }));
};

// Helper function to check if model is loaded
export const isModelLoaded = (modelId: string, modelsLoadedStore: Record<string, boolean>): boolean => {
  const actualModelId = resolveActualModelId(modelId, []); // We'll resolve this in the calling function
  return modelsLoadedStore[actualModelId] ?? false;
};

// FIXED: Main function to wake up a model with proper deduplication
export const wakeUpModel = async (
  modelId: string,
  models: Model[],
  i18n: I18n,
  retryCount: number = 0
): Promise<boolean> => {
  if (!modelId || !models.length) return false;

  const MAX_RETRIES = 3;
  const RETRY_DELAY = 30000; // 30 seconds

  try {
    const actualModelId = resolveActualModelId(modelId, models);
    const now = Date.now();

    // CRITICAL FIX 1: Check if model is already marked as loaded
    const currentModelsState = get(modelsLoaded);
    if (currentModelsState[actualModelId] === true) {
      console.log(`Model ${actualModelId} is already loaded, skipping wake-up call`);
      return true;
    }

    // CRITICAL FIX 2: Prevent duplicate wake-up calls
    if (wakingUpModels.has(actualModelId)) {
      console.log(`Model ${actualModelId} is already being woken up, skipping duplicate call`);
      return false;
    }

    // CRITICAL FIX 3: Check cooldown period
    const lastAttempt = recentWakeupAttempts.get(actualModelId) || 0;
    if (now - lastAttempt < WAKEUP_COOLDOWN) {
      console.log(`Model ${actualModelId} was attempted recently (${Math.round((now - lastAttempt)/1000)}s ago), skipping`);
      return false;
    }

    console.log(`Waking up model: ${modelId} -> actual: ${actualModelId} (attempt ${retryCount + 1})`);

    // Mark this model as being processed
    wakingUpModels.add(actualModelId);
    recentWakeupAttempts.set(actualModelId, now);

    // Set model as loading initially
    setModelLoaded(actualModelId, false);

    // CRITICAL FIX 4: Add unique timestamp to prevent any caching issues
    const timestamp = Date.now();
    const response = await fetch(`${WEBUI_API_BASE_URL}/utils/wake_up_models?t=${timestamp}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.token}`,
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      },
      body: JSON.stringify({
        chat_model: actualModelId,
        embedding_model: "Linq-AI-Research/Linq-Embed-Mistral",
        reranker_model: "BAAI/bge-reranker-v2-m3",
        force: false
      }),
      cache: 'no-store'
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result: WakeUpResponse = await response.json();
    console.log('Model wake up response:', result);

    // CRITICAL FIX 5: Properly handle the server responses
    if (result.status === "All models already awake" ||
        result.status === "All models successfully awakened") {
      setModelLoaded(actualModelId, true);
      wakingUpModels.delete(actualModelId);
      console.log(`Model ${actualModelId} confirmed as loaded by server`);
      return true;
    }

    // Check chat model status specifically
    const chatModelStatus = result.chat_model?.status;
    const chatModelSuccess = result.chat_model?.success;

    if (chatModelStatus) {
      // New server implementation with status field
      if (chatModelStatus === 'awake') {
        setModelLoaded(actualModelId, true);
        wakingUpModels.delete(actualModelId);

        if (result.chat_model?.needs_wakeup && retryCount === 0) {
          toast.success(i18n.t('Model {{modelName}} is ready', { modelName: modelId }));
        }
        return true;

      } else if (chatModelStatus === 'loading') {
        // Model is being loaded by another process
        setModelLoaded(actualModelId, false);

        if (retryCount === 0) {
          toast.info(i18n.t('Model {{modelName}} is being loaded...', { modelName: modelId }));
        }

        // Wait and retry once after 5 seconds, but don't retry indefinitely
        if (retryCount < 2) {
          setTimeout(() => {
            wakingUpModels.delete(actualModelId);
            wakeUpModel(modelId, models, i18n, retryCount + 1);
          }, 5000);
        } else {
          wakingUpModels.delete(actualModelId);
        }
        return false;

      } else if (chatModelStatus === 'failed') {
        setModelLoaded(actualModelId, false);
        wakingUpModels.delete(actualModelId);

        if (retryCount < MAX_RETRIES) {
          console.log(`Model ${actualModelId} failed, retrying in ${RETRY_DELAY/1000}s (attempt ${retryCount + 1}/${MAX_RETRIES})`);
          setTimeout(() => {
            wakeUpModel(modelId, models, i18n, retryCount + 1);
          }, RETRY_DELAY);
        } else {
          toast.error(i18n.t('Failed to load model {{modelName}} after {{retries}} attempts', {
            modelName: modelId,
            retries: MAX_RETRIES
          }));
        }
        return false;

      } else if (chatModelStatus === 'unavailable') {
        setModelLoaded(actualModelId, false);
        wakingUpModels.delete(actualModelId);
        if (retryCount === 0) {
          toast.warning(i18n.t('Model {{modelName}} is not available', { modelName: modelId }));
        }
        return false;
      }
    } else if (typeof chatModelSuccess === 'boolean') {
      // Old server implementation - check success field
      if (chatModelSuccess) {
        setModelLoaded(actualModelId, true);
        wakingUpModels.delete(actualModelId);

        if (result.chat_model?.needs_wakeup && retryCount === 0) {
          toast.success(i18n.t('Model {{modelName}} is ready', { modelName: modelId }));
        }
        return true;
      } else {
        setModelLoaded(actualModelId, false);
        wakingUpModels.delete(actualModelId);

        if (retryCount < MAX_RETRIES) {
          setTimeout(() => {
            wakeUpModel(modelId, models, i18n, retryCount + 1);
          }, RETRY_DELAY);
        } else {
          toast.error(i18n.t('Failed to load model {{modelName}}', { modelName: modelId }));
        }
        return false;
      }
    }

    // If we get here, the response format is unexpected
    console.warn('Unexpected response format:', result);
    setModelLoaded(actualModelId, false);
    wakingUpModels.delete(actualModelId);
    return false;

  } catch (error) {
    console.error('Error waking up model:', error);
    const actualModelId = resolveActualModelId(modelId, models);
    wakingUpModels.delete(actualModelId);
    setModelLoaded(actualModelId, false);

    if (retryCount < MAX_RETRIES) {
      console.log(`Network error, retrying in ${RETRY_DELAY/1000}s (attempt ${retryCount + 1}/${MAX_RETRIES})`);
      setTimeout(() => {
        wakeUpModel(modelId, models, i18n, retryCount + 1);
      }, RETRY_DELAY);
    } else {
      if (retryCount === 0) {
        toast.error(i18n.t('Network error loading model {{modelName}}', { modelName: modelId }));
      }
    }

    return false;
  }
};

// Add this to modelWakeup.ts after your existing functions

// Track last message time
export const lastModelInteractionTimes = writable<Record<string, number>>({});

export const updateLastInteractionTime = (modelId: string, models: Model[]) => {
  const actualModelId = resolveActualModelId(modelId, models);
  const now = Date.now();

  lastModelInteractionTimes.update(currentTimes => {
    return {
      ...currentTimes,
      [actualModelId]: now
    };
  });
  console.log(`Updated last interaction time for ${actualModelId} to ${new Date(now).toLocaleString()}`);
};

export const needsModelCheck = (modelId: string, models: Model[]): boolean => {
  const actualModelId = resolveActualModelId(modelId, models);
  let needsCheck = true; // Default to true

  lastModelInteractionTimes.subscribe(currentTimes => {
    const lastTime = currentTimes[actualModelId];

    if (lastTime) {
      const elapsed = Date.now() - lastTime;
      // If it has been less than 15 minutes, no check is needed.
      if (elapsed < 15 * 60 * 1000) {
        needsCheck = false;
      }
    }
  })(); // Immediately execute and unsubscribe

  console.log(`Checking if model ${actualModelId} needs a wake-up call: ${needsCheck}`);
  return needsCheck;
};

// Update the ensureModelsAwakeSSE function signature to add a quiet option:
export const ensureModelsAwakeSSE = async (
  chatModel: string,
  models: Model[],
  i18n: I18n,
  options: { quiet?: boolean; skipTimeCheck?: boolean } = {}
): Promise<boolean> => {
  const { quiet = false, skipTimeCheck = false } = options;
  const actualModelId = resolveActualModelId(chatModel, models);

  // If less than 15 minutes and not skipping time check, assume models are still awake
  if (!skipTimeCheck && !needsModelCheck(chatModel, models)) {
    console.log('Less than 15 minutes since last message, skipping check');
    return true;
  }

  return new Promise((resolve, reject) => {
    let loadingToast: any = null;

    fetch(`${WEBUI_API_BASE_URL}/utils/wake_up_models_sse`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        chat_model: actualModelId,
        embedding_model: "Linq-AI-Research/Linq-Embed-Mistral",
        reranker_model: "BAAI/bge-reranker-v2-m3",
        force: false
      })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const processStream = async () => {
        while (true) {
          const { done, value } = await reader.read();

          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                switch (data.type) {
                  case 'acknowledged':
                    console.log('Server acknowledged request:', data.message);
                    if (!quiet) {
                      loadingToast = toast.loading(i18n.t('Checking model status...'));
                    }
                    break;

                  case 'status':
                    console.log('Status update:', data);
                    if (!quiet) {
                      const chatStatus = data.models.chat_model?.status;
                      const embeddingStatus = data.models.embedding_model?.status;
                      const rerankerStatus = data.models.reranker_model?.status;

                      const loadingModels = [];
                      if (chatStatus === 'loading') loadingModels.push('Chat');
                      if (embeddingStatus === 'loading') loadingModels.push('Embedding');
                      if (rerankerStatus === 'loading') loadingModels.push('Reranker');

                      if (loadingModels.length > 0) {
                        if (loadingToast) toast.dismiss(loadingToast);
                        loadingToast = toast.loading(i18n.t('Loading models: {{models}}', {
                          models: loadingModels.join(', ')
                        }));
                      }
                    }
                    break;

                  case 'complete':
                    console.log('All models ready!', data);
                    if (!quiet && loadingToast) toast.dismiss(loadingToast);
                    if (!quiet) toast.success(i18n.t('Models are ready!'));
                    setModelLoaded(actualModelId, true);

                    updateLastInteractionTime(actualModelId, models);

                    resolve(true);
                    return;

                  case 'timeout':
                    console.log('Timeout reached:', data);
                    if (!quiet && loadingToast) toast.dismiss(loadingToast);
                    if (!quiet) toast.error(i18n.t('Models took too long to load'));
                    resolve(false);
                    return;

                  case 'error':
                    console.error('Error:', data);
                    if (!quiet && loadingToast) toast.dismiss(loadingToast);
                    if (!quiet) toast.error(i18n.t('Error: {{message}}', { message: data.message }));
                    resolve(false);
                    return;
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            }
          }
        }
      };

      processStream().catch(error => {
        console.error('Stream processing error:', error);
        if (!quiet && loadingToast) toast.dismiss(loadingToast);
        if (!quiet) toast.error(i18n.t('Connection error'));
        resolve(false);
      });
    })
    .catch(error => {
      console.error('Fetch error:', error);
      if (!quiet && loadingToast) toast.dismiss(loadingToast);
      if (!quiet) toast.error(i18n.t('Failed to connect to server'));
      resolve(false);
    });
  });
};

// Function to check if selected models are loaded
export const areSelectedModelsLoaded = (
  selectedModels: string[],
  atSelectedModel: { id: string } | undefined,
  modelsLoadedStore: Record<string, boolean>,
  models: Model[]
): boolean => {
  // If using atSelectedModel (@ mention)
  if (atSelectedModel) {
    const actualModelId = resolveActualModelId(atSelectedModel.id, models);
    const isLoaded = modelsLoadedStore[actualModelId] ?? false;
    console.log(`Checking atSelectedModel ${atSelectedModel.id} -> ${actualModelId}: ${isLoaded}`);
    return isLoaded;
  }

  // If no models selected
  if (!selectedModels || selectedModels.length === 0 || (selectedModels.length === 1 && selectedModels[0] === '')) {
    console.log('No models selected');
    return false;
  }

  // Check only the currently selected models
  const allLoaded = selectedModels.every(modelId => {
    if (!modelId || modelId === '') return true; // Skip empty selections

    const actualModelId = resolveActualModelId(modelId, models);
    const isLoaded = modelsLoadedStore[actualModelId] ?? false;
    console.log(`Checking selectedModel ${modelId} -> ${actualModelId}: ${isLoaded}`);
    return isLoaded;
  });

  console.log('All selected models loaded:', allLoaded);
  return allLoaded;
};

// Cleanup function to remove old models from modelsLoaded
export const cleanupModelsLoaded = (selectedModels: string[], models: Model[]): void => {
  const currentSelectedIds = selectedModels
    .filter(id => id && id !== '')
    .map(id => resolveActualModelId(id, models));

  modelsLoaded.update(current => {
    const cleaned: Record<string, boolean> = {};
    // Keep currently selected models
    Object.keys(current).forEach(modelId => {
      if (currentSelectedIds.includes(modelId)) {
        cleaned[modelId] = current[modelId];
      }
    });
    return cleaned;
  });
};

// Helper function to check if a model needs wake-up (considering cooldown)
export const shouldWakeUpModel = (modelId: string, models: Model[]): boolean => {
  const actualModelId = resolveActualModelId(modelId, models);
  const currentState = get(modelsLoaded);

  // Don't wake up if already loaded
  if (currentState[actualModelId] === true) {
    return false;
  }

  // Don't wake up if currently being woken up
  if (wakingUpModels.has(actualModelId)) {
    return false;
  }

  // Don't wake up if recently attempted
  const lastAttempt = recentWakeupAttempts.get(actualModelId) || 0;
  const now = Date.now();
  if (now - lastAttempt < WAKEUP_COOLDOWN) {
    return false;
  }

  return true;
};