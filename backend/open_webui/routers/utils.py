import json

import black
import logging
import markdown
import requests

from open_webui.models.chats import ChatTitleMessagesForm
from open_webui.config import DATA_DIR, ENABLE_ADMIN_EXPORT
from open_webui.constants import ERROR_MESSAGES
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel
from starlette.responses import FileResponse


from open_webui.utils.misc import get_gravatar_url
from open_webui.utils.pdf_generator import PDFGenerator
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.code_interpreter import execute_code_jupyter
from open_webui.env import SRC_LOG_LEVELS


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])

router = APIRouter()


@router.get("/gravatar")
async def get_gravatar(email: str, user=Depends(get_verified_user)):
    return get_gravatar_url(email)


class CodeForm(BaseModel):
    code: str


@router.post("/code/format")
async def format_code(form_data: CodeForm, user=Depends(get_verified_user)):
    try:
        formatted_code = black.format_str(form_data.code, mode=black.Mode())
        return {"code": formatted_code}
    except black.NothingChanged:
        return {"code": form_data.code}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/code/execute")
async def execute_code(
    request: Request, form_data: CodeForm, user=Depends(get_verified_user)
):
    if request.app.state.config.CODE_EXECUTION_ENGINE == "jupyter":
        output = await execute_code_jupyter(
            request.app.state.config.CODE_EXECUTION_JUPYTER_URL,
            form_data.code,
            (
                request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH_TOKEN
                if request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH == "token"
                else None
            ),
            (
                request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH_PASSWORD
                if request.app.state.config.CODE_EXECUTION_JUPYTER_AUTH == "password"
                else None
            ),
            request.app.state.config.CODE_EXECUTION_JUPYTER_TIMEOUT,
        )

        return output
    else:
        raise HTTPException(
            status_code=400,
            detail="Code execution engine not supported",
        )


class MarkdownForm(BaseModel):
    md: str


@router.post("/markdown")
async def get_html_from_markdown(
    form_data: MarkdownForm, user=Depends(get_verified_user)
):
    return {"html": markdown.markdown(form_data.md)}


class ChatForm(BaseModel):
    title: str
    messages: list[dict]


@router.post("/pdf")
async def download_chat_as_pdf(
    form_data: ChatTitleMessagesForm, user=Depends(get_verified_user)
):
    try:
        pdf_bytes = PDFGenerator(form_data).generate_chat_pdf()

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment;filename=chat.pdf"},
        )
    except Exception as e:
        log.exception(f"Error generating PDF: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/db/download")
async def download_db(user=Depends(get_admin_user)):
    if not ENABLE_ADMIN_EXPORT:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    from open_webui.internal.db import engine

    if engine.name != "sqlite":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DB_NOT_SQLITE,
        )
    return FileResponse(
        engine.url.database,
        media_type="application/octet-stream",
        filename="webui.db",
    )


@router.get("/litellm/config")
async def download_litellm_config_yaml(user=Depends(get_admin_user)):
    return FileResponse(
        f"{DATA_DIR}/litellm/config.yaml",
        media_type="application/octet-stream",
        filename="config.yaml",
    )

############ My code ############

import json
import time
import os
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter


api_key = os.getenv("OPENAI_API_KEY")
API_ENDPOINT = os.getenv("OPENAI_API_BASE_URL", "https://inference-dev.rcp.epfl.ch/v1")

# Last wake-up timestamp tracker
last_wakeup_time = 0  # Unix timestamp of the last successful wake-up
WAKEUP_INTERVAL = 15 * 60  # 15 minutes in seconds

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

from typing import Optional, Dict
from pydantic import BaseModel


# Global dictionary to track last wake-up time per model
last_wakeup_times: Dict[str, float] = {}


# Add this model class for the request body
class WakeUpModelsRequest(BaseModel):
    force: Optional[bool] = False
    embedding_model: Optional[str] = "Linq-AI-Research/Linq-Embed-Mistral"
    chat_model: Optional[str] = "Qwen/Qwen3-30B-A3B"





# Global dictionary to track active wake-up tasks per model
active_wakeup_tasks: Dict[str, asyncio.Task] = {}

# Lock to prevent race conditions
wakeup_lock = asyncio.Lock()


async def check_model_availability_async(model_id):
    """
    Check if a model is available in the inference provider by querying the /models endpoint
    """
    log.info(f"[AVAILABILITY] Checking availability for model: {model_id}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                    f"{API_ENDPOINT}/models",
                    headers=headers,
                    timeout=30
            ) as response:
                response.raise_for_status()
                models_data = await response.json()

                available_models = [model.get('id') for model in models_data.get("data", [])]
                log.info(f"[AVAILABILITY] Available models: {available_models}")

                is_available = model_id in available_models
                log.info(f"[AVAILABILITY] Model {model_id} availability: {is_available}")

                return is_available

        except Exception as e:
            log.error(f"[AVAILABILITY] Error checking model availability for {model_id}: {e}")
            return True


@router.post("/wake_up_models")
async def wake_up_models(request: WakeUpModelsRequest):
    """
    Wake up models by concurrently sending requests to embedding and chat completion endpoints.
    This endpoint handles concurrent requests properly - multiple users can wake up different models simultaneously.
    """
    global last_wakeup_times, active_wakeup_tasks
    current_time = time.time()

    log.info(
        f"[WAKE_UP] New wake-up request received - Embedding: {request.embedding_model}, Chat: {request.chat_model}")
    log.info(f"[WAKE_UP] Current active tasks: {list(active_wakeup_tasks.keys())}")
    log.info(f"[WAKE_UP] Current asyncio task queue size: {len(asyncio.all_tasks())}")

    # Check if these specific models are already being woken up
    embedding_task_active = request.embedding_model in active_wakeup_tasks
    chat_task_active = request.chat_model in active_wakeup_tasks

    if embedding_task_active:
        log.info(f"[WAKE_UP] Embedding model {request.embedding_model} is already being woken up by another request")
    if chat_task_active:
        log.info(f"[WAKE_UP] Chat model {request.chat_model} is already being woken up by another request")

    # First check if models are available in the inference provider
    log.info(f"[WAKE_UP] Checking model availability...")
    embedding_available = await check_model_availability_async(request.embedding_model)
    chat_available = await check_model_availability_async(request.chat_model)

    # If models are not available, consider them as "awake" (skip wake-up)
    if not embedding_available:
        log.info(
            f"[WAKE_UP] Embedding model {request.embedding_model} not available in inference provider, considering as awake")

    if not chat_available:
        log.info(f"[WAKE_UP] Chat model {request.chat_model} not available in inference provider, considering as awake")

    # Check wake-up status for each available model
    embedding_needs_wakeup = embedding_available and not embedding_task_active
    chat_needs_wakeup = chat_available and not chat_task_active

    embedding_last_wakeup = last_wakeup_times.get(request.embedding_model, 0)
    chat_last_wakeup = last_wakeup_times.get(request.chat_model, 0)

    embedding_elapsed = current_time - embedding_last_wakeup
    chat_elapsed = current_time - chat_last_wakeup

    # Check if embedding model needs wake-up (only if available and not already being woken up)
    if embedding_available and not request.force and embedding_last_wakeup > 0 and embedding_elapsed < WAKEUP_INTERVAL:
        embedding_needs_wakeup = False
        embedding_minutes_ago = int(embedding_elapsed / 60)
        embedding_next_wakeup = int((WAKEUP_INTERVAL - embedding_elapsed) / 60)
        log.info(
            f"[WAKE_UP] Embedding model {request.embedding_model} was awakened {embedding_minutes_ago} minutes ago, skipping")

    # Check if chat model needs wake-up (only if available and not already being woken up)
    if chat_available and not request.force and chat_last_wakeup > 0 and chat_elapsed < WAKEUP_INTERVAL:
        chat_needs_wakeup = False
        chat_minutes_ago = int(chat_elapsed / 60)
        chat_next_wakeup = int((WAKEUP_INTERVAL - chat_elapsed) / 60)
        log.info(f"[WAKE_UP] Chat model {request.chat_model} was awakened {chat_minutes_ago} minutes ago, skipping")

    log.info(f"[WAKE_UP] Wake-up decision - Embedding: {embedding_needs_wakeup}, Chat: {chat_needs_wakeup}")

    # If neither model needs wake-up, return early
    if not embedding_needs_wakeup and not chat_needs_wakeup:
        log.info(f"[WAKE_UP] No models need wake-up, returning early")
        return {
            "status": "Models already awake",
            "embedding_model": {
                "name": request.embedding_model,
                "last_wakeup": f"{embedding_minutes_ago if embedding_available and not embedding_task_active else 'N/A'} minutes ago" if embedding_available else "N/A (not available)",
                "next_wakeup_in": f"{embedding_next_wakeup} minutes" if embedding_available and not embedding_task_active else "N/A",
                "needs_wakeup": False,
                "success": True,
                "available": embedding_available,
                "task_active": embedding_task_active
            },
            "chat_model": {
                "name": request.chat_model,
                "last_wakeup": f"{chat_minutes_ago if chat_available and not chat_task_active else 'N/A'} minutes ago" if chat_available else "N/A (not available)",
                "next_wakeup_in": f"{chat_next_wakeup} minutes" if chat_available and not chat_task_active else "N/A",
                "needs_wakeup": False,
                "success": True,
                "available": chat_available,
                "task_active": chat_task_active
            }
        }

    log.info(
        f"[WAKE_UP] Starting model wake-up process for models - Embedding: {request.embedding_model} (available: {embedding_available}, needs_wakeup: {embedding_needs_wakeup}), Chat: {request.chat_model} (available: {chat_available}, needs_wakeup: {chat_needs_wakeup})")

    try:
        # Initialize results - models not available are considered successful
        embedding_success = not embedding_available or not embedding_needs_wakeup
        chat_success = not chat_available or not chat_needs_wakeup

        # Create list of tasks to run (only for available models that need wake-up)
        tasks = []
        task_types = []
        task_models = []

        if embedding_available and embedding_needs_wakeup:
            log.info(f"[WAKE_UP] Creating embedding wake-up task for {request.embedding_model}")
            task = asyncio.create_task(get_embeddings_async('a', model=request.embedding_model))
            active_wakeup_tasks[request.embedding_model] = task
            tasks.append(task)
            task_types.append('embedding')
            task_models.append(request.embedding_model)

        if chat_available and chat_needs_wakeup:
            log.info(f"[WAKE_UP] Creating chat wake-up task for {request.chat_model}")
            task = asyncio.create_task(test_chat_completion_async(model=request.chat_model))
            active_wakeup_tasks[request.chat_model] = task
            tasks.append(task)
            task_types.append('chat')
            task_models.append(request.chat_model)

        log.info(f"[WAKE_UP] Active tasks after creation: {list(active_wakeup_tasks.keys())}")
        log.info(f"[WAKE_UP] Total asyncio tasks in event loop: {len(asyncio.all_tasks())}")

        # Only run tasks if there are any to run
        if tasks:
            log.info(f"[WAKE_UP] Starting {len(tasks)} concurrent wake-up tasks")
            start_time = time.time()

            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            log.info(f"[WAKE_UP] All wake-up tasks completed in {end_time - start_time:.2f} seconds")

            # Clean up active tasks
            for model in task_models:
                if model in active_wakeup_tasks:
                    del active_wakeup_tasks[model]
                    log.info(f"[WAKE_UP] Removed {model} from active tasks")

            # Process results
            result_index = 0
            for i, task_type in enumerate(task_types):
                result = results[result_index]
                result_index += 1

                if task_type == 'embedding':
                    if isinstance(result, Exception):
                        log.error(f"[WAKE_UP] Embedding task failed: {result}")
                        embedding_success = False
                    else:
                        embedding_success = result and result.get("data") is not None
                        log.info(f"[WAKE_UP] Embedding task completed successfully: {embedding_success}")

                elif task_type == 'chat':
                    if isinstance(result, Exception):
                        log.error(f"[WAKE_UP] Chat task failed: {result}")
                        chat_success = False
                    else:
                        chat_success = bool(result)
                        log.info(f"[WAKE_UP] Chat task completed successfully: {chat_success}")

        # Update last wake-up times for successfully awakened models
        if embedding_success and embedding_available and embedding_needs_wakeup:
            last_wakeup_times[request.embedding_model] = current_time
            log.info(f"[WAKE_UP] Updated last_wakeup_time for {request.embedding_model}")

        if chat_success and chat_available and chat_needs_wakeup:
            last_wakeup_times[request.chat_model] = current_time
            log.info(f"[WAKE_UP] Updated last_wakeup_time for {request.chat_model}")

        log.info(f"[WAKE_UP] Final active tasks: {list(active_wakeup_tasks.keys())}")
        log.info(f"[WAKE_UP] Request completed - Embedding success: {embedding_success}, Chat success: {chat_success}")

        # Prepare detailed response
        response = {
            "embedding_model": {
                "name": request.embedding_model,
                "success": embedding_success,
                "needed_wakeup": embedding_needs_wakeup,
                "available": embedding_available,
                "task_was_active": embedding_task_active,
                "last_wakeup": "0 minutes ago" if embedding_available and embedding_needs_wakeup and embedding_success else (
                    f"{int(embedding_elapsed / 60)} minutes ago" if embedding_available else "N/A (not available)"),
                "timestamp": current_time if embedding_available and embedding_needs_wakeup and embedding_success else (
                    embedding_last_wakeup if embedding_available else None)
            },
            "chat_model": {
                "name": request.chat_model,
                "success": chat_success,
                "needed_wakeup": chat_needs_wakeup,
                "available": chat_available,
                "task_was_active": chat_task_active,
                "last_wakeup": "0 minutes ago" if chat_available and chat_needs_wakeup and chat_success else (
                    f"{int(chat_elapsed / 60)} minutes ago" if chat_available else "N/A (not available)"),
                "timestamp": current_time if chat_available and chat_needs_wakeup and chat_success else (
                    chat_last_wakeup if chat_available else None)
            }
        }

        # Determine overall status
        if embedding_success and chat_success:
            response["status"] = "All models are awake"
        elif embedding_success:
            response["status"] = "Embedding model is awake, but chat model failed to wake up"
        elif chat_success:
            response["status"] = "Chat model is awake, but embedding model failed to wake up"
        else:
            response["status"] = "Failed to wake up models"

        return response

    except Exception as e:
        log.error(f"[WAKE_UP] Error waking up models: {str(e)}")
        import traceback
        log.error(f"[WAKE_UP] Traceback: {traceback.format_exc()}")

        # Clean up active tasks in case of error
        for model in [request.embedding_model, request.chat_model]:
            if model in active_wakeup_tasks:
                del active_wakeup_tasks[model]
                log.info(f"[WAKE_UP] Cleaned up failed task for {model}")

        return {
            "status": "Error",
            "message": str(e),
            "embedding_model": {
                "name": request.embedding_model,
                "success": False,
                "needed_wakeup": embedding_needs_wakeup,
                "available": embedding_available
            },
            "chat_model": {
                "name": request.chat_model,
                "success": False,
                "needed_wakeup": chat_needs_wakeup,
                "available": chat_available
            }
        }

def check_model_availability_sync(model_id):
    """
    Synchronous version to check if a model is available in the inference provider

    Args:
        model_id (str): The model ID to check for availability

    Returns:
        bool: True if model is available, False otherwise
    """
    log.info(f"Checking availability for model: {model_id}")

    try:
        response = requests.get(f"{API_ENDPOINT}/models", headers=headers, timeout=30)
        response.raise_for_status()

        models_data = response.json()
        available_models = [model.get('id') for model in models_data.get("data", [])]
        log.info(f"Available models: {available_models}")

        is_available = model_id in available_models
        log.info(f"Model {model_id} availability: {is_available}")

        return is_available

    except requests.exceptions.RequestException as e:
        log.error(f"Error checking model availability for {model_id}: {e}")
        # If we can't check availability, assume model is available to avoid blocking
        return True

async def test_chat_completion_async(model="Qwen/Qwen3-30B-A3B"):
    """
    Asynchronous version of test_chat_completion
    """
    log.info("Testing chat completion asynchronously...")
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant utile."},
            {"role": "user", "content": "Bonjour, pouvez-vous me dire quel jour nous sommes?"}
        ],
        "temperature": 0.7,
        "max_tokens": 5,
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                    f"{API_ENDPOINT}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=180
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    return True
                return False

        except Exception as e:  # Changed from aiohttp.ClientError to Exception
            log.error(f"Chat completion error: {e}")
            return False


async def get_embeddings_async(text, model="Linq-AI-Research/Linq-Embed-Mistral"):
    """
    Asynchronous version of get_embeddings
    """
    log.info("Getting embeddings asynchronously...")
    # Ensure text is a list if it's a single string
    if isinstance(text, str):
        text = [text]

    # Prepare request payload
    payload = {
        "model": model,
        "input": text
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                    f"{API_ENDPOINT}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=180
            ) as response:
                response.raise_for_status()
                return await response.json()

        except Exception as e:  # Changed from aiohttp.ClientError to Exception
            log.error(f"Embedding error: {e}")
            return None

def get_embeddings(text, model="Linq-AI-Research/Linq-Embed-Mistral"):
    """
    Get embeddings for the provided text using the specified model.

    Args:
        text (str or list): The input text to embed. Can be a single string or a list of strings.
        model (str): The embedding model to use. Default is "Linq-AI-Research/Linq-Embed-Mistral".

    Returns:
        dict: The API response containing the embeddings.
    """
    # API endpoint
    base_url = API_ENDPOINT
    endpoint = f"{base_url}/embeddings"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Ensure text is a list if it's a single string
    if isinstance(text, str):
        text = [text]

    # Prepare request payload
    payload = {
        "model": model,
        "input": text
    }

    # Make the request
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=180
        )

        # Check if the request was successful
        response.raise_for_status()

        # Return the response data
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

def test_chat_completion(model="Qwen/Qwen3-30B-A3B"):
    """Teste l'endpoint /chat/completions avec un modèle spécifié"""
    print(f"\nTest de l'endpoint /chat/completions avec le modèle {model}...")

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant utile."},
            {"role": "user", "content": "Bonjour, pouvez-vous me dire quel jour nous sommes?"}
        ],
        "temperature": 0.7,
        "max_tokens": 5,
    }

    try:
        response = requests.post(f"{API_ENDPOINT}/chat/completions", headers=headers, json=data)
        response.raise_for_status()

        print(f"Statut: {response.status_code}")
        result = response.json()

        # Affichage formaté de la réponse
        print("Réponse:")
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            content = message.get("content", "")
            print(f"[{message.get('role', 'assistant')}]: {content}")
        else:
            print("Pas de réponse dans le format attendu")
            print(json.dumps(result, indent=2))

        return True
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête à l'endpoint /chat/completions: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Détails de l'erreur: {e.response.text}")
        return False
