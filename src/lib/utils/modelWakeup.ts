import { get } from 'svelte/store';
import { toast } from 'svelte-sonner';
import { modelsLoaded } from '$lib/stores';
import { WEBUI_API_BASE_URL } from '$lib/constants';

// Type definitions
interface Model {
  id: string;
  info?: {
    base_model_id?: string;
    [key: string]: any; // Allow other properties
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
//
// CHANGE: Use a Map to store in-flight promises instead of a Set of active IDs.
// This is the core of the fix.
//
const activeWakeupPromises = new Map<string, Promise<boolean>>();

/**
 * Resolve the actual model ID (handles base models)
 */
export const resolveActualModelId = (modelId: string, models: Model[]): string => {
  const model = models.find(m => m.id === modelId);
  const actualId = model?.info?.base_model_id || modelId;
  // Reduced logging to avoid console spam
  if (modelId !== actualId) {
    console.log(`Resolving model ID: ${modelId} -> ${actualId}`);
  }
  return actualId;
};

/**
 * Create minimal models_info object with only necessary fields
 */
const createMinimalModelsInfo = (modelId: string, models: Model[]): Record<string, { info: { base_model_id?: string } }> => {
  const model = models.find(m => m.id === modelId);
  const minimalInfo = model && model.info?.base_model_id ? { base_model_id: model.info.base_model_id } : {};
  return { [modelId]: { info: minimalInfo } };
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
    const minimalModelsInfo = models ? createMinimalModelsInfo(modelId, models) : {};
    const response = await fetch(
      `${WEBUI_API_BASE_URL}/utils/check_model_availability/${encodeURIComponent(modelId)}?models_info=${encodeURIComponent(JSON.stringify(minimalModelsInfo))}`,
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
  return true; // Assume available if check fails
};

/**
 * Wake up models using SSE - REFACTORED to be non-blocking
 */
// export const ensureModelsAwakeSSE = async (
//   chatModel: string,
//   models: Model[],
//   i18n: I18n,
//   options: { quiet?: boolean } = {}
// ): Promise<boolean> => {
//   const { quiet = false } = options;
//   const actualModelId = resolveActualModelId(chatModel, models);

//   // 1. Check if model is already loaded in the store
//   if (get(modelsLoaded)[actualModelId] === true) {
//     return true;
//   }

//   // 2. CHANGE: Check if a wake-up promise already exists for this model.
//   // If so, return the existing promise instead of creating a new one.
//   if (activeWakeupPromises.has(actualModelId)) {
//     console.log(`A wake-up call for ${actualModelId} is already in progress. Awaiting its completion.`);
//     return activeWakeupPromises.get(actualModelId)!;
//   }

//   // 3. Create a new promise for the wake-up process.
//   const wakeupPromise = new Promise<boolean>((resolve) => {
//     let loadingToast: any = null;

//     const cleanupAndResolve = (result: boolean) => {
//       if (loadingToast && !quiet) toast.dismiss(loadingToast);
//       resolve(result);
//     };

//     const timeout = setTimeout(() => {
//         console.error(`SSE timeout reached for model ${actualModelId}`);
//         if (!quiet) {
//             if (loadingToast) toast.dismiss(loadingToast);
//             toast.error(i18n.t('Model loading timed out.'));
//         }
//         cleanupAndResolve(false);
//     }, SSE_TIMEOUT);

//     const minimalModelsInfo = createMinimalModelsInfo(chatModel, models);
//     console.log(`Starting new wake-up call for ${actualModelId}`);

//     fetch(`${WEBUI_API_BASE_URL}/utils/wake_up_models_sse`, {
//       method: 'POST',
//       headers: {
//         'Authorization': `Bearer ${localStorage.token}`,
//         'Content-Type': 'application/json',
//         'Cache-Control': 'no-cache'
//       },
//       body: JSON.stringify({
//         chat_model: chatModel,
//         models_info: minimalModelsInfo
//       })
//     })
//     .then(response => {
//       if (!response.ok) {
//         throw new Error(`HTTP error ${response.status}`);
//       }
//       const reader = response.body?.getReader();
//       if (!reader) {
//         throw new Error('No response body reader');
//       }
//       const decoder = new TextDecoder();
//       let buffer = '';

//       const processStream = async () => {
//         while (true) {
//           const { done, value } = await reader.read();
//           if (done) {
//             console.warn(`SSE stream for ${actualModelId} ended without a 'complete' event.`);
//             break;
//           }
//           buffer += decoder.decode(value, { stream: true });
//           const lines = buffer.split('\n');
//           buffer = lines.pop() || '';
//           for (const line of lines) {
//             if (line.startsWith('data: ')) {
//               try {
//                 const data = JSON.parse(line.slice(6));
//                 switch (data.type) {
//                   case 'acknowledged':
//                     if (!quiet) loadingToast = toast.loading(i18n.t('Checking model status...'));
//                     break;
//                   case 'status':
//                     if (!quiet && data.models?.chat_model?.status === 'loading') {
//                       if (loadingToast) toast.dismiss(loadingToast);
//                       loadingToast = toast.loading(i18n.t('Loading model...'));
//                     }
//                     break;
//                   case 'complete':
//                     clearTimeout(timeout);
//                     if (!quiet) {
//                       if (loadingToast) toast.dismiss(loadingToast);
//                       toast.success(i18n.t('Model is ready!'));
//                     }
//                     modelsLoaded.update(state => ({ ...state, [actualModelId]: true }));
//                     updateLastInteractionTime(actualModelId, models);
//                     cleanupAndResolve(true);
//                     return; // Exit processing loop
//                   case 'timeout':
//                   case 'error':
//                     clearTimeout(timeout);
//                     if (!quiet) {
//                       if (loadingToast) toast.dismiss(loadingToast);
//                       toast.error(data.message || i18n.t('Failed to load model'));
//                     }
//                     cleanupAndResolve(false);
//                     return; // Exit processing loop
//                 }
//               } catch (e) { /* Ignore parsing errors for malformed SSE data */ }
//             }
//           }
//         }
//       };
//       processStream().catch(error => {
//         clearTimeout(timeout);
//         console.error('SSE stream processing error:', error);
//         cleanupAndResolve(false);
//       });
//     })
//     .catch(error => {
//       clearTimeout(timeout);
//       console.error('SSE fetch request failed:', error);
//       if (!quiet) toast.error(i18n.t('Failed to connect to server'));
//       cleanupAndResolve(false);
//     });
//   });

//   // 4. Store the new promise in the map.
//   activeWakeupPromises.set(actualModelId, wakeupPromise);

//   // 5. Use .finally() to ensure the promise is removed from the map
//   //    whether it resolves to true or false.
//   wakeupPromise.finally(() => {
//     activeWakeupPromises.delete(actualModelId);
//   });

//   // 6. Return the promise to the caller.
//   return wakeupPromise;
// };

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

  // Check the new map for an in-progress promise
  if (activeWakeupPromises.has(actualModelId)) return false;

  return true;
};

/**
 * Wake up a specific model
 */
// export const wakeUpModel = async (
//   modelId: string,
//   models: Model[],
//   i18n: I18n
// ): Promise<boolean> => {
//   if (!modelId || !models.length) return false;

//   const actualModelId = resolveActualModelId(modelId, models);

//   // Check if available from provider
//   const isAvailable = await checkModelAvailability(modelId, models);
//   if (!isAvailable) {
//     console.log(`Model ${modelId} is not available from provider`);
//     return true; // Consider unavailable models as "loaded"
//   }

//   // Use SSE wake-up with minimal data
//   return await ensureModelsAwakeSSE(modelId, models, i18n);
// };