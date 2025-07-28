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
from fastapi import APIRouter, Response
import requests

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

# Setup logging
log = logging.getLogger(__name__)

# Global dictionary to track last wake-up time per model
last_wakeup_times: Dict[str, float] = {}


# Add this model class for the request body
class WakeUpModelsRequest(BaseModel):
    force: Optional[bool] = False
    embedding_model: Optional[str] = "Linq-AI-Research/Linq-Embed-Mistral"
    chat_model: Optional[str] = "Qwen/Qwen3-30B-A3B"
    reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3"


# Global dictionary to track active wake-up tasks per model
active_wakeup_tasks: Dict[str, asyncio.Task] = {}

# Lock to prevent race conditions
wakeup_lock = asyncio.Lock()

# Add task timeout for safety
TASK_TIMEOUT = 1800  # 30 minutes max per task


def cleanup_finished_tasks():
    """Clean up any finished, cancelled, or failed tasks from active_wakeup_tasks"""
    models_to_remove = []

    for model, task in active_wakeup_tasks.items():
        if task.done():  # Task is finished (completed, cancelled, or failed)
            models_to_remove.append(model)
            if task.cancelled():
                log.warning(f"[CLEANUP] Task for model {model} was cancelled")
            elif task.exception():
                log.error(f"[CLEANUP] Task for model {model} failed with exception: {task.exception()}")
            else:
                log.info(f"[CLEANUP] Task for model {model} completed successfully")

    for model in models_to_remove:
        del active_wakeup_tasks[model]

    return len(models_to_remove)


async def safe_task_cleanup(task_models: list):
    """Safely clean up tasks for specific models, ensuring they're removed even if tasks fail"""
    for model in task_models:
        if model in active_wakeup_tasks:
            task = active_wakeup_tasks[model]
            if not task.done():
                try:
                    task.cancel()
                    # Wait a bit for cancellation to complete
                    try:
                        await asyncio.wait_for(task, timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass  # Expected for cancelled tasks
                except Exception as e:
                    log.error(f"[CLEANUP] Error cancelling task for {model}: {e}")

            # Always remove from active tasks
            del active_wakeup_tasks[model]
            log.info(f"[CLEANUP] Removed task for model {model}")


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
                is_available = model_id in available_models
                log.info(f"[AVAILABILITY] Model {model_id} availability: {is_available}")

                return is_available

        except Exception as e:
            log.error(f"[AVAILABILITY] Error checking model availability for {model_id}: {e}")
            return True


@router.post("/wake_up_models")
async def wake_up_models(request: WakeUpModelsRequest, response: Response):
    """
    Wake up models by concurrently sending requests to embedding, chat completion, and reranker endpoints.
    This endpoint handles concurrent requests properly - multiple users can wake up different models simultaneously.
    """
    # Set cache control headers to prevent caching
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    global last_wakeup_times, active_wakeup_tasks
    current_time = time.time()

    log.info(
        f"[WAKE_UP] New wake-up request received - Embedding: {request.embedding_model}, "
        f"Chat: {request.chat_model}, Reranker: {request.reranker_model}"
    )

    # Clean up any finished tasks first
    cleaned_count = cleanup_finished_tasks()
    if cleaned_count > 0:
        log.info(f"[WAKE_UP] Cleaned up {cleaned_count} finished tasks")

    log.info(f"[WAKE_UP] Current active tasks: {list(active_wakeup_tasks.keys())}")

    # Check if these specific models are already being woken up
    embedding_task_active = request.embedding_model in active_wakeup_tasks
    chat_task_active = request.chat_model in active_wakeup_tasks
    reranker_task_active = request.reranker_model in active_wakeup_tasks

    # First check if models are available in the inference provider
    log.info(f"[WAKE_UP] Checking model availability...")
    embedding_available = await check_model_availability_async(request.embedding_model)
    chat_available = await check_model_availability_async(request.chat_model)
    reranker_available = await check_model_availability_async(request.reranker_model)

    # Get last wake-up times
    embedding_last_wakeup = last_wakeup_times.get(request.embedding_model, 0)
    chat_last_wakeup = last_wakeup_times.get(request.chat_model, 0)
    reranker_last_wakeup = last_wakeup_times.get(request.reranker_model, 0)

    embedding_elapsed = current_time - embedding_last_wakeup
    chat_elapsed = current_time - chat_last_wakeup
    reranker_elapsed = current_time - reranker_last_wakeup

    log.info(
        f"[WAKE_UP] Elapsed time since last wake-up - "
        f"Embedding: {embedding_elapsed}, Chat: {chat_elapsed}, Reranker: {reranker_elapsed} "
        f"for models {request.embedding_model}, {request.chat_model}, and {request.reranker_model}"
    )

    # Determine the status for each model
    embedding_status = get_model_status(
        available=embedding_available,
        task_active=embedding_task_active,
        last_wakeup=embedding_last_wakeup,
        elapsed=embedding_elapsed,
        force=request.force
    )

    chat_status = get_model_status(
        available=chat_available,
        task_active=chat_task_active,
        last_wakeup=chat_last_wakeup,
        elapsed=chat_elapsed,
        force=request.force
    )

    reranker_status = get_model_status(
        available=reranker_available,
        task_active=reranker_task_active,
        last_wakeup=reranker_last_wakeup,
        elapsed=reranker_elapsed,
        force=request.force
    )

    # Check if any model is currently being loaded
    if embedding_task_active or chat_task_active or reranker_task_active:
        log.info(f"[WAKE_UP] Models are being loaded by another process")

        result = {
            "status": "Models are being loaded",
            "embedding_model": {
                "name": request.embedding_model,
                "available": embedding_available,
                "task_active": embedding_task_active,
                "needs_wakeup": False,
                "success": False,
                "status": "loading" if embedding_task_active else embedding_status,
                "last_wakeup": format_last_wakeup(embedding_elapsed, embedding_available, embedding_task_active)
            },
            "chat_model": {
                "name": request.chat_model,
                "available": chat_available,
                "task_active": chat_task_active,
                "needs_wakeup": False,
                "success": False,
                "status": "loading" if chat_task_active else chat_status,
                "last_wakeup": format_last_wakeup(chat_elapsed, chat_available, chat_task_active)
            },
            "reranker_model": {
                "name": request.reranker_model,
                "available": reranker_available,
                "task_active": reranker_task_active,
                "needs_wakeup": False,
                "success": False,
                "status": "loading" if reranker_task_active else reranker_status,
                "last_wakeup": format_last_wakeup(reranker_elapsed, reranker_available, reranker_task_active)
            }
        }
        return result

    # Check if models need wake-up
    embedding_needs_wakeup = (
            embedding_available and
            not embedding_task_active and
            (request.force or embedding_last_wakeup == 0 or embedding_elapsed >= WAKEUP_INTERVAL)
    )

    chat_needs_wakeup = (
            chat_available and
            not chat_task_active and
            (request.force or chat_last_wakeup == 0 or chat_elapsed >= WAKEUP_INTERVAL)
    )

    reranker_needs_wakeup = (
            reranker_available and
            not reranker_task_active and
            (request.force or reranker_last_wakeup == 0 or reranker_elapsed >= WAKEUP_INTERVAL)
    )

    # If no model needs wake-up, they are all awake
    if not embedding_needs_wakeup and not chat_needs_wakeup and not reranker_needs_wakeup:
        log.info(f"[WAKE_UP] All models are already awake")

        result = {
            "status": "All models already awake",
            "embedding_model": {
                "name": request.embedding_model,
                "available": embedding_available,
                "task_active": False,
                "needs_wakeup": False,
                "success": True,
                "status": "awake" if embedding_available else "unavailable",
                "last_wakeup": format_last_wakeup(embedding_elapsed, embedding_available, False),
                "next_wakeup_in": format_next_wakeup(embedding_elapsed) if embedding_available else "N/A"
            },
            "chat_model": {
                "name": request.chat_model,
                "available": chat_available,
                "task_active": False,
                "needs_wakeup": False,
                "success": True,
                "status": "awake" if chat_available else "unavailable",
                "last_wakeup": format_last_wakeup(chat_elapsed, chat_available, False),
                "next_wakeup_in": format_next_wakeup(chat_elapsed) if chat_available else "N/A"
            },
            "reranker_model": {
                "name": request.reranker_model,
                "available": reranker_available,
                "task_active": False,
                "needs_wakeup": False,
                "success": True,
                "status": "awake" if reranker_available else "unavailable",
                "last_wakeup": format_last_wakeup(reranker_elapsed, reranker_available, False),
                "next_wakeup_in": format_next_wakeup(reranker_elapsed) if reranker_available else "N/A"
            }
        }
        return result

    # If we reach here, we need to wake up at least one model
    log.info(
        f"[WAKE_UP] Starting wake-up process - "
        f"Embedding: {embedding_needs_wakeup}, Chat: {chat_needs_wakeup}, Reranker: {reranker_needs_wakeup}"
    )

    # Initialize task tracking
    tasks = []
    task_types = []
    task_models = []

    try:
        # Initialize results
        embedding_success = not embedding_needs_wakeup
        chat_success = not chat_needs_wakeup
        reranker_success = not reranker_needs_wakeup

        # Create tasks for models that need wake-up with timeout wrapper
        if embedding_needs_wakeup:
            log.info(f"[WAKE_UP] Creating embedding wake-up task for {request.embedding_model}")
            task = asyncio.create_task(
                asyncio.wait_for(
                    get_embeddings_async('wake up test', model=request.embedding_model),
                    timeout=TASK_TIMEOUT
                )
            )
            active_wakeup_tasks[request.embedding_model] = task
            tasks.append(task)
            task_types.append('embedding')
            task_models.append(request.embedding_model)

        if chat_needs_wakeup:
            log.info(f"[WAKE_UP] Creating chat wake-up task for {request.chat_model}")
            task = asyncio.create_task(
                asyncio.wait_for(
                    test_chat_completion_async(model=request.chat_model),
                    timeout=TASK_TIMEOUT
                )
            )
            active_wakeup_tasks[request.chat_model] = task
            tasks.append(task)
            task_types.append('chat')
            task_models.append(request.chat_model)

        if reranker_needs_wakeup:
            log.info(f"[WAKE_UP] Creating reranker wake-up task for {request.reranker_model}")
            task = asyncio.create_task(
                asyncio.wait_for(
                    test_reranker_async(model=request.reranker_model),
                    timeout=TASK_TIMEOUT
                )
            )
            active_wakeup_tasks[request.reranker_model] = task
            tasks.append(task)
            task_types.append('reranker')
            task_models.append(request.reranker_model)

        # Run tasks if any
        if tasks:
            log.info(f"[WAKE_UP] Running {len(tasks)} wake-up tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for i, (task_type, result) in enumerate(zip(task_types, results)):
                if task_type == 'embedding':
                    if isinstance(result, Exception):
                        log.error(f"[WAKE_UP] Embedding task failed: {result}")
                        embedding_success = False
                    else:
                        embedding_success = result and result.get("data") is not None
                        if embedding_success:
                            last_wakeup_times[request.embedding_model] = current_time

                elif task_type == 'chat':
                    if isinstance(result, Exception):
                        log.error(f"[WAKE_UP] Chat task failed: {result}")
                        chat_success = False
                    else:
                        chat_success = bool(result)
                        if chat_success:
                            last_wakeup_times[request.chat_model] = current_time

                elif task_type == 'reranker':
                    if isinstance(result, Exception):
                        log.error(f"[WAKE_UP] Reranker task failed: {result}")
                        reranker_success = False
                    else:
                        reranker_success = bool(result)
                        if reranker_success:
                            last_wakeup_times[request.reranker_model] = current_time

        # Prepare final response
        result = {
            "embedding_model": {
                "name": request.embedding_model,
                "available": embedding_available,
                "task_active": False,
                "needs_wakeup": embedding_needs_wakeup,
                "success": embedding_success,
                "status": get_final_status(embedding_available, embedding_needs_wakeup, embedding_success),
                "last_wakeup": format_last_wakeup(
                    0 if (embedding_needs_wakeup and embedding_success) else embedding_elapsed,
                    embedding_available,
                    False
                )
            },
            "chat_model": {
                "name": request.chat_model,
                "available": chat_available,
                "task_active": False,
                "needs_wakeup": chat_needs_wakeup,
                "success": chat_success,
                "status": get_final_status(chat_available, chat_needs_wakeup, chat_success),
                "last_wakeup": format_last_wakeup(
                    0 if (chat_needs_wakeup and chat_success) else chat_elapsed,
                    chat_available,
                    False
                )
            },
            "reranker_model": {
                "name": request.reranker_model,
                "available": reranker_available,
                "task_active": False,
                "needs_wakeup": reranker_needs_wakeup,
                "success": reranker_success,
                "status": get_final_status(reranker_available, reranker_needs_wakeup, reranker_success),
                "last_wakeup": format_last_wakeup(
                    0 if (reranker_needs_wakeup and reranker_success) else reranker_elapsed,
                    reranker_available,
                    False
                )
            }
        }

        # Set overall status
        all_successful = embedding_success and chat_success and reranker_success
        no_wakeup_needed = not embedding_needs_wakeup and not chat_needs_wakeup and not reranker_needs_wakeup
        partial_success = embedding_success or chat_success or reranker_success

        if all_successful:
            result["status"] = "All models successfully awakened"
        elif no_wakeup_needed:
            result["status"] = "All models already awake"
        elif partial_success:
            result["status"] = "Partial success"
        else:
            result["status"] = "Failed to wake up models"

        return result

    except Exception as e:
        log.error(f"[WAKE_UP] Error: {str(e)}")
        return {
            "status": "Error",
            "error": str(e),
            "embedding_model": {
                "name": request.embedding_model,
                "success": False,
                "status": "error"
            },
            "chat_model": {
                "name": request.chat_model,
                "success": False,
                "status": "error"
            },
            "reranker_model": {
                "name": request.reranker_model,
                "success": False,
                "status": "error"
            }
        }

    finally:
        # Always clean up tasks in the finally block
        await safe_task_cleanup(task_models)


# Add these imports at the top of utils.py if not already present
from fastapi.responses import StreamingResponse
import asyncio
import json


# Add this new SSE endpoint after your existing wake_up_models function
@router.post("/wake_up_models_sse")
async def wake_up_models_sse(request: WakeUpModelsRequest):
    """
    Wake up models and stream status updates via Server-Sent Events
    """

    async def event_generator():
        try:
            # Send initial acknowledgment
            yield f"data: {json.dumps({'type': 'acknowledged', 'message': 'Request received, checking models...'})}\n\n"

            max_wait_time = 600  # 10 minutes
            update_interval = 10  # Send update every 10 seconds
            start_time = time.time()
            last_update_time = start_time

            # Initial wake-up call
            initial_result = await wake_up_models(request, Response())

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'models': initial_result})}\n\n"

            # Check if all models are already awake
            all_awake = (
                    initial_result.get("chat_model", {}).get("status") == "awake" and
                    initial_result.get("embedding_model", {}).get("status") == "awake" and
                    initial_result.get("reranker_model", {}).get("status") == "awake"
            )

            if all_awake:
                yield f"data: {json.dumps({'type': 'complete', 'message': 'All models are ready!', 'models': initial_result})}\n\n"
                return

            # Wait for models to be ready
            while time.time() - start_time < max_wait_time:
                await asyncio.sleep(3)  # Check every 3 seconds internally

                # Only send update every 10 seconds
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    # Check current status
                    current_status = await wake_up_models(request, Response())

                    # Send status update
                    yield f"data: {json.dumps({'type': 'status', 'models': current_status, 'elapsed': int(current_time - start_time)})}\n\n"
                    last_update_time = current_time

                    # Check if all models are ready
                    all_ready = (
                            current_status.get("chat_model", {}).get("status") == "awake" and
                            current_status.get("embedding_model", {}).get("status") == "awake" and
                            current_status.get("reranker_model", {}).get("status") == "awake"
                    )

                    if all_ready:
                        yield f"data: {json.dumps({'type': 'complete', 'message': 'All models are ready!', 'models': current_status})}\n\n"
                        return

            # Timeout
            final_status = await wake_up_models(request, Response())
            yield f"data: {json.dumps({'type': 'timeout', 'message': 'Timeout reached, some models may still be loading', 'models': final_status})}\n\n"

        except Exception as e:
            log.error(f"Error in SSE stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )

# Helper functions (unchanged)
def get_model_status(available, task_active, last_wakeup, elapsed, force):
    """Determine the current status of a model"""
    if not available:
        return "unavailable"
    if task_active:
        return "loading"
    if force or last_wakeup == 0:
        return "needs_wakeup"
    if elapsed < WAKEUP_INTERVAL:
        return "awake"
    return "needs_wakeup"


def format_last_wakeup(elapsed, available, task_active):
    """Format the last wake-up time for display"""
    if not available:
        return "N/A (not available)"
    if task_active:
        return "Currently loading"
    if elapsed == 0:
        return "Just awakened"
    minutes = int(elapsed / 60)
    return f"{minutes} minutes ago"


def format_next_wakeup(elapsed):
    """Format the next wake-up time"""
    remaining = WAKEUP_INTERVAL - elapsed
    if remaining <= 0:
        return "Now"
    minutes = int(remaining / 60)
    return f"{minutes} minutes"


def get_final_status(available, needed_wakeup, success):
    """Get the final status after wake-up attempt"""
    if not available:
        return "unavailable"
    if needed_wakeup:
        return "awake" if success else "failed"
    return "awake"


def check_model_availability_sync(model_id):
    """
    Synchronous version to check if a model is available in the inference provider
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
        return True


async def test_reranker_async(model="BAAI/bge-reranker-v2-m3"):
    """
    Asynchronous test for reranker model to wake it up - with cache disabled
    """
    log.info(f"Testing reranker model {model} asynchronously...")

    # Make each request unique to prevent any caching issues
    timestamp = int(time.time() * 1000)  # millisecond timestamp
    unique_id = f"wakeup_{timestamp}"

    # Simple test data for wake-up with unique content
    data = {
        "model": model,
        "query": f"wake up test query {unique_id}",
        "documents": [
            f"This is a test document for model wake-up {unique_id}.",
            f"Another test document to ensure the reranker is loaded {unique_id}."
        ],
        "top_k": 2,
        # Disable caching to ensure the request reaches the model
        "cache": {
            "no-cache": True,
            "no-store": True
        }
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                    f"{API_ENDPOINT}/rerank",
                    headers=headers,
                    json=data,
                    timeout=TASK_TIMEOUT
            ) as response:
                response.raise_for_status()
                result = await response.json()

                # Check if we got a valid reranking response
                if "results" in result and len(result["results"]) > 0:
                    log.info(f"Reranker {model} successfully awakened")
                    return True

                log.warning(f"Reranker {model} responded but with unexpected format")
                return False

        except Exception as e:
            log.error(f"Reranker wake-up error for {model}: {e}")
            return False


async def test_chat_completion_async(model="Qwen/Qwen3-30B-A3B"):
    """
    Asynchronous version of test_chat_completion - with cache disabled
    """
    log.info(f"[WAKE UP] Testing chat completion asynchronously... for model: {model}")

    # Make each request unique to prevent caching
    timestamp = int(time.time() * 1000)
    unique_id = f"wakeup_{timestamp}"

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant utile."},
            {"role": "user", "content": f"Bonjour, test de réveil du modèle {unique_id}"}
        ],
        "temperature": 0.7,
        "max_tokens": 5,
        # Disable caching to ensure the request reaches the model
        "cache": {
            "no-cache": True,
            "no-store": True
        }
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                    f"{API_ENDPOINT}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=TASK_TIMEOUT
            ) as response:
                response.raise_for_status()
                result = await response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    log.info(f"[WAKE UP] Chat completion successful for model {model}")
                    return True
                return False

        except Exception as e:
            log.error(f"Chat completion error: {e}")
            return False


async def get_embeddings_async(text, model="Linq-AI-Research/Linq-Embed-Mistral"):
    """
    Asynchronous version of get_embeddings - with cache disabled
    """
    log.info("Getting embeddings asynchronously...")

    # For wake-up calls, make the text unique
    if isinstance(text, str) and text == 'wake up test':
        timestamp = int(time.time() * 1000)
        text = f'wake up test {timestamp}'

    if isinstance(text, str):
        text = [text]

    payload = {
        "model": model,
        "input": text,
        # Disable caching to ensure the request reaches the model
        "cache": {
            "no-cache": True,
            "no-store": True
        }
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                    f"{API_ENDPOINT}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=TASK_TIMEOUT
            ) as response:
                response.raise_for_status()
                return await response.json()

        except Exception as e:
            log.error(f"Embedding error: {e}")
            return None


# Also update the synchronous versions for consistency
def get_embeddings(text, model="Linq-AI-Research/Linq-Embed-Mistral"):
    """
    Get embeddings for the provided text using the specified model - with cache disabled
    """
    base_url = API_ENDPOINT
    endpoint = f"{base_url}/embeddings"

    headers_with_auth = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # For wake-up calls, make the text unique
    if isinstance(text, str) and text == 'wake up test':
        timestamp = int(time.time() * 1000)
        text = f'wake up test {timestamp}'

    if isinstance(text, str):
        text = [text]

    payload = {
        "model": model,
        "input": text,
        # Disable caching to ensure the request reaches the model
        "cache": {
            "no-cache": True,
            "no-store": True
        }
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers_with_auth,
            data=json.dumps(payload),
            timeout=TASK_TIMEOUT
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None


def test_chat_completion(model="Qwen/Qwen3-30B-A3B"):
    """Synchronous test for chat completion - with cache disabled"""
    print(f"\nTest de l'endpoint /chat/completions avec le modèle {model}...")

    # Make each request unique to prevent caching
    timestamp = int(time.time() * 1000)
    unique_id = f"wakeup_{timestamp}"

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant utile."},
            {"role": "user", "content": f"Bonjour, test de réveil du modèle {unique_id}"}
        ],
        "temperature": 0.7,
        "max_tokens": 5,
        # Disable caching to ensure the request reaches the model
        "cache": {
            "no-cache": True,
            "no-store": True
        }
    }

    try:
        response = requests.post(f"{API_ENDPOINT}/chat/completions", headers=headers, json=data)
        response.raise_for_status()

        print(f"Statut: {response.status_code}")
        result = response.json()

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