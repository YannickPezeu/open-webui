// modelWakeup.ts - Optimized version with minimal data transfer

import { get } from 'svelte/store';
import { toast } from 'svelte-sonner';
import { modelsLoaded } from '$lib/stores';
import { WEBUI_API_BASE_URL } from '$lib/constants';

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

// Constants
const WAKEUP_INTERVAL = 15 * 60 * 1000; // 15 minutes in milliseconds
const SSE_TIMEOUT = 600 * 1000; // 10 minutes

// State management
const modelLastActivity = new Map<string, number>();
const activeWakeups = new Set<string>();

/**
 * Resolve the actual model ID (handles base models)
 */
export const resolveActualModelId = (modelId: string, models: Model[]): string => {
  const model = models.find(m => m.id === modelId);
  const actualId = model?.info?.base_model_id || modelId;
  console.log(`Resolving model ID: ${modelId} -> ${actualId}`);
  return actualId;
};

/**
 * Update the last activity time for a model
 */
export const updateLastInteractionTime = (modelId: string, models: Model[]) => {
  const actualModelId = resolveActualModelId(modelId, models);
  modelLastActivity.set(actualModelId, Date.now());
};

/**
 * Check if a model needs to be woken up
 */
export const needsModelCheck = (modelId: string, models: Model[]): boolean => {
  const actualModelId = resolveActualModelId(modelId, models);
  const lastActivity = modelLastActivity.get(actualModelId) || 0;
  return Date.now() - lastActivity > WAKEUP_INTERVAL;
};

/**
 * Check if a model is available from the provider
 */
export const checkModelAvailability = async (modelId: string, models?: Model[]): Promise<boolean> => {
  try {
    // Only send info for the specific model being checked, not all models
    const model = models?.find(m => m.id === modelId);
    const modelInfo = model ? { [modelId]: { info: model.info || {} } } : {};

    const response = await fetch(
      `${WEBUI_API_BASE_URL}/utils/check_model_availability/${encodeURIComponent(modelId)}?models_info=${encodeURIComponent(JSON.stringify(modelInfo))}`,
      {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.token}`,
          'Content-Type': 'application/json'
        }
      }
    );

    if (response.ok) {
      const data = await response.json();
      return data.available;
    }
  } catch (error) {
    console.error(`Error checking model availability for ${modelId}:`, error);
  }

  // If we can't check, assume available to not block users
  return true;
};

/**
 * Wake up models using SSE - RESTORED with minimal data
 */
export const ensureModelsAwakeSSE = async (
  chatModel: string,
  models: Model[],
  i18n: I18n,
  options: { quiet?: boolean } = {}
): Promise<boolean> => {
  const { quiet = false } = options;
  const actualModelId = resolveActualModelId(chatModel, models);

  // Check if already waking up
  if (activeWakeups.has(actualModelId)) {
    console.log(`Model ${actualModelId} is already being woken up`);
    return false;
  }

  // Check if model is already loaded
  const currentState = get(modelsLoaded);
  if (currentState[actualModelId] === true) {
    console.log(`Model ${actualModelId} is already loaded`);
    return true;
  }

  // Mark as being woken up
  activeWakeups.add(actualModelId);

  try {
    return await new Promise((resolve) => {
      let loadingToast: any = null;
      let resolved = false;

      const cleanup = () => {
        activeWakeups.delete(actualModelId);
        if (loadingToast && !quiet) toast.dismiss(loadingToast);
        if (!resolved) {
          resolved = true;
          resolve(false);
        }
      };

      // Set a timeout
      const timeout = setTimeout(() => {
        console.error('SSE timeout reached');
        cleanup();
      }, SSE_TIMEOUT);

      // Create minimal models info - ONLY for the specific model being woken up
      const model = models.find(m => m.id === chatModel);
      const minimalModelsInfo = model ? { [chatModel]: { info: model.info || {} } } : {};

      console.log('Minimal models_info size:', JSON.stringify(minimalModelsInfo).length, 'chars');

      // Create SSE request with minimal data
      fetch(`${WEBUI_API_BASE_URL}/utils/wake_up_models_sse`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.token}`,
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache'
        },
        body: JSON.stringify({
          chat_model: chatModel,
          embedding_model: "Linq-AI-Research/Linq-Embed-Mistral",
          reranker_model: "BAAI/bge-reranker-v2-m3",
          force: false,
          models_info: minimalModelsInfo  // ← Only the specific model, not all models!
        })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('No response body');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        const processStream = async () => {
          try {
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
                        if (!quiet) {
                          loadingToast = toast.loading(i18n.t('Checking model status...'));
                        }
                        break;

                      case 'status':
                        if (!quiet && data.models?.chat_model?.status === 'loading') {
                          if (loadingToast) toast.dismiss(loadingToast);
                          loadingToast = toast.loading(i18n.t('Loading model...'));
                        }
                        break;

                      case 'complete':
                        clearTimeout(timeout);
                        if (!quiet) {
                          if (loadingToast) toast.dismiss(loadingToast);
                          toast.success(i18n.t('Model is ready!'));
                        }
                        modelsLoaded.update(state => ({ ...state, [actualModelId]: true }));
                        updateLastInteractionTime(actualModelId, models);
                        activeWakeups.delete(actualModelId);
                        resolved = true;
                        resolve(true);
                        return;

                      case 'timeout':
                      case 'error':
                        clearTimeout(timeout);
                        if (!quiet) {
                          if (loadingToast) toast.dismiss(loadingToast);
                          toast.error(data.message || i18n.t('Failed to load model'));
                        }
                        cleanup();
                        return;
                    }
                  } catch (e) {
                    console.error('Error parsing SSE data:', e);
                  }
                }
              }
            }
          } catch (error) {
            console.error('Stream processing error:', error);
            cleanup();
          }
        };

        processStream();
      })
      .catch(error => {
        clearTimeout(timeout);
        console.error('SSE request failed:', error);
        if (!quiet) toast.error(i18n.t('Failed to connect to server'));
        cleanup();
      });
    });
  } catch (error) {
    activeWakeups.delete(actualModelId);
    throw error;
  }
};

/**
 * Check if selected models are loaded
 */
export const areSelectedModelsLoaded = async (
  selectedModels: string[],
  atSelectedModel: { id: string } | undefined,
  modelsLoadedStore: Record<string, boolean>,
  models: Model[]
): Promise<boolean> => {
  // If using @mention
  if (atSelectedModel) {
    const actualModelId = resolveActualModelId(atSelectedModel.id, models);

    // Check availability - if not available from provider, consider it "loaded"
    const isAvailable = await checkModelAvailability(atSelectedModel.id, models);
    if (!isAvailable) return true;

    return modelsLoadedStore[actualModelId] ?? false;
  }

  // Check all selected models
  const validModels = selectedModels.filter(id => id && id !== '');
  if (validModels.length === 0) return false;

  for (const modelId of validModels) {
    const actualModelId = resolveActualModelId(modelId, models);

    // Check availability
    const isAvailable = await checkModelAvailability(modelId, models);
    if (!isAvailable) continue; // Skip unavailable models

    // Check if loaded
    if (!modelsLoadedStore[actualModelId]) {
      return false;
    }
  }

  return true;
};

/**
 * Clean up old models from modelsLoaded store
 */
export const cleanupModelsLoaded = (selectedModels: string[], models: Model[]): void => {
  const currentSelectedIds = selectedModels
    .filter(id => id && id !== '')
    .map(id => resolveActualModelId(id, models));

  modelsLoaded.update(current => {
    const cleaned: Record<string, boolean> = {};
    Object.keys(current).forEach(modelId => {
      if (currentSelectedIds.includes(modelId)) {
        cleaned[modelId] = current[modelId];
      }
    });
    return cleaned;
  });
};

/**
 * Check if a model should be woken up
 */
export const shouldWakeUpModel = (modelId: string, models: Model[]): boolean => {
  const actualModelId = resolveActualModelId(modelId, models);
  const currentState = get(modelsLoaded);

  // Already loaded
  if (currentState[actualModelId] === true) return false;

  // Already being woken up
  if (activeWakeups.has(actualModelId)) return false;

  return true;
};

/**
 * Wake up a specific model
 */
export const wakeUpModel = async (
  modelId: string,
  models: Model[],
  i18n: I18n
): Promise<boolean> => {
  if (!modelId || !models.length) return false;

  const actualModelId = resolveActualModelId(modelId, models);

  // Check if available from provider
  const isAvailable = await checkModelAvailability(modelId, models);
  if (!isAvailable) {
    console.log(`Model ${modelId} is not available from provider`);
    return true; // Consider unavailable models as "loaded"
  }

  // Use SSE wake-up with minimal data
  return await ensureModelsAwakeSSE(modelId, models, i18n);
};