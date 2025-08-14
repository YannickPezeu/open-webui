import json
import black
import logging
import markdown
import requests
import time
import os
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional, Dict, Set
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.responses import FileResponse

from open_webui.models.chats import ChatTitleMessagesForm
from open_webui.config import DATA_DIR, ENABLE_ADMIN_EXPORT
from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.misc import get_gravatar_url
from open_webui.utils.pdf_generator import PDFGenerator
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.code_interpreter import execute_code_jupyter
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])

router = APIRouter()


# Existing endpoints
@router.get("/gravatar")
async def get_gravatar(email: str, user=Depends(get_verified_user)):
    return get_gravatar_url(email)


class CodeForm(BaseModel):
    code: str


@router.post("/code/format")
async def format_code(form_data: CodeForm, user=Depends(get_admin_user)):
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


# ============ Optimized Model Wake-up System ============

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
API_ENDPOINT = os.getenv("OPENAI_API_BASE_URL", "https://inference-dev.rcp.epfl.ch/v1")
WAKEUP_INTERVAL = 15 * 60  # 15 minutes in seconds
TASK_TIMEOUT = 300  # 5 minutes max per wake-up task

# Global state
model_last_activity: Dict[str, float] = {}
active_wakeup_tasks: Set[str] = set()
wakeup_lock = asyncio.Lock()


class WakeUpModelsRequest(BaseModel):
    chat_model: str
    embedding_model: Optional[str] = "Linq-AI-Research/Linq-Embed-Mistral"
    reranker_model: Optional[str] = "BAAI/bge-reranker-v2-m3"
    force: Optional[bool] = False
    models_info: Optional[dict] = {}  # Now minimal - only specific model info


@router.post("/wake_up_models_sse")
async def wake_up_models_sse(request: WakeUpModelsRequest, user=Depends(get_verified_user)):
    """
    Wake up models and stream status updates via Server-Sent Events.
    Optimized to handle minimal models_info payload.
    """

    async def event_generator():
        try:
            # Log payload size for monitoring
            payload_size = len(json.dumps(request.models_info))
            log.info(f"SSE Wake-up request - models_info size: {payload_size} chars")

            yield f"data: {json.dumps({'type': 'acknowledged', 'message': 'Starting model check...'})}\n\n"

            # Resolve actual model IDs using minimal info
            actual_chat_model = resolve_actual_model_id(request.chat_model, request.models_info)
            actual_embedding_model = resolve_actual_model_id(request.embedding_model, request.models_info)
            actual_reranker_model = resolve_actual_model_id(request.reranker_model, request.models_info)

            log.info(f"Model ID mapping: {request.chat_model} -> {actual_chat_model}")

            # Check which models need wake-up
            models_to_wake = await get_models_needing_wakeup(
                actual_chat_model,
                actual_embedding_model,
                actual_reranker_model,
                request.force
            )

            if not models_to_wake:
                yield f"data: {json.dumps({'type': 'complete', 'message': 'All models ready', 'models': {'chat_model': {'status': 'awake'}, 'embedding_model': {'status': 'awake'}, 'reranker_model': {'status': 'awake'}}})}\n\n"
                return

            # Wake up models that need it
            async with wakeup_lock:
                tasks = []

                if 'chat' in models_to_wake:
                    if actual_chat_model not in active_wakeup_tasks:
                        active_wakeup_tasks.add(actual_chat_model)
                        task = asyncio.create_task(wake_chat_model(actual_chat_model))
                        tasks.append(('chat', actual_chat_model, task))

                if 'embedding' in models_to_wake:
                    if actual_embedding_model not in active_wakeup_tasks:
                        active_wakeup_tasks.add(actual_embedding_model)
                        task = asyncio.create_task(wake_embedding_model(actual_embedding_model))
                        tasks.append(('embedding', actual_embedding_model, task))

                if 'reranker' in models_to_wake:
                    if actual_reranker_model not in active_wakeup_tasks:
                        active_wakeup_tasks.add(actual_reranker_model)
                        task = asyncio.create_task(wake_reranker_model(actual_reranker_model))
                        tasks.append(('reranker', actual_reranker_model, task))

            # Monitor wake-up progress
            start_time = time.time()
            max_wait = 600  # 10 minutes

            while tasks and time.time() - start_time < max_wait:
                status = {}
                all_ready = True

                for task_type, model_id, task in tasks:
                    if task.done():
                        try:
                            result = task.result()
                            status[f"{task_type}_model"] = {
                                'name': model_id,
                                'status': 'awake' if result else 'failed'
                            }
                        except Exception as e:
                            status[f"{task_type}_model"] = {
                                'name': model_id,
                                'status': 'failed',
                                'error': str(e)
                            }
                    else:
                        status[f"{task_type}_model"] = {
                            'name': model_id,
                            'status': 'loading'
                        }
                        all_ready = False

                yield f"data: {json.dumps({'type': 'status', 'models': status, 'elapsed': int(time.time() - start_time)})}\n\n"

                if all_ready:
                    # Clean up completed tasks
                    for _, model_id, _ in tasks:
                        active_wakeup_tasks.discard(model_id)

                    yield f"data: {json.dumps({'type': 'complete', 'message': 'All models ready', 'models': status})}\n\n"
                    return

                await asyncio.sleep(3)

            # Timeout
            for _, model_id, _ in tasks:
                active_wakeup_tasks.discard(model_id)

            yield f"data: {json.dumps({'type': 'timeout', 'message': 'Some models may still be loading'})}\n\n"

        except Exception as e:
            log.error(f"SSE error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/wake_up_models")
async def wake_up_models_simple(request: WakeUpModelsRequest, user=Depends(get_verified_user)):
    """Simple non-SSE wake-up endpoint (fallback)."""

    payload_size = len(json.dumps(request.models_info))
    log.info(f"Simple wake-up request - models_info size: {payload_size} chars")

    # Resolve actual model IDs
    actual_chat_model = resolve_actual_model_id(request.chat_model, request.models_info)
    log.info(f"Simple wake-up for model: {request.chat_model} -> {actual_chat_model}")

    # Check if models need wake-up
    models_to_wake = await get_models_needing_wakeup(
        actual_chat_model,
        request.embedding_model or "Linq-AI-Research/Linq-Embed-Mistral",
        request.reranker_model or "BAAI/bge-reranker-v2-m3",
        request.force or False
    )

    if not models_to_wake:
        return {"success": True, "message": "All models ready"}

    # Wake up chat model if needed
    if 'chat' in models_to_wake:
        success = await wake_chat_model(actual_chat_model)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to wake up model")

    return {"success": True, "message": "Model awakened"}


@router.get("/check_model_availability/{model_id:path}")
async def check_model_availability(
        model_id: str,
        user=Depends(get_verified_user),
        models_info: Optional[str] = None
):
    """Check if a specific model is available from the provider."""

    # Parse minimal models_info if provided
    models_dict = {}
    if models_info:
        try:
            models_dict = json.loads(models_info)
            log.info(f"Availability check - models_info size: {len(models_info)} chars")
        except:
            pass

    # Resolve actual model ID
    actual_model_id = resolve_actual_model_id(model_id, models_dict)

    is_available = await is_model_available(actual_model_id)
    return {
        "available": is_available,
        "model_id": model_id,
        "actual_model_id": actual_model_id
    }


# Helper functions (unchanged)
async def get_available_models() -> Set[str]:
    """Get list of available models from provider."""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_ENDPOINT}/models", headers=headers, timeout=30) as response:
                if response.status == 200:
                    data = await response.json()
                    return {model.get('id') for model in data.get('data', [])}
    except Exception as e:
        log.error(f"Failed to get available models: {e}")

    return set()


async def is_model_available(model_id: str) -> bool:
    """Check if a model is available from the provider."""
    available_models = await get_available_models()
    return model_id in available_models


def resolve_actual_model_id(model_id: str, models_info: dict) -> str:
    """
    Resolve the actual model ID from OpenWebUI's model ID.
    Now works with minimal models_info containing only specific model.
    """
    if model_id in models_info:
        model = models_info[model_id]
        if isinstance(model, dict) and 'info' in model:
            base_model_id = model['info'].get('base_model_id')
            if base_model_id:
                log.info(f"Resolved {model_id} -> {base_model_id} via models_info")
                return base_model_id

    # Otherwise, return the original ID
    return model_id


async def get_models_needing_wakeup(chat_model: str, embedding_model: str, reranker_model: str, force: bool) -> Set[
    str]:
    """Determine which models need wake-up."""
    current_time = time.time()
    models_to_wake = set()

    # Get available models
    available_models = await get_available_models()

    # Check each model
    for model_type, model_id in [('chat', chat_model), ('embedding', embedding_model), ('reranker', reranker_model)]:
        if model_id in available_models:
            last_activity = model_last_activity.get(model_id, 0)
            if force or current_time - last_activity > WAKEUP_INTERVAL:
                models_to_wake.add(model_type)

    return models_to_wake


async def wake_chat_model(model_id: str) -> bool:
    """Wake up a chat model."""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": "wake up"}],
            "max_tokens": 1,
            "temperature": 0
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{API_ENDPOINT}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=TASK_TIMEOUT
            ) as response:
                if response.status == 200:
                    model_last_activity[model_id] = time.time()
                    log.info(f"Chat model {model_id} awakened")
                    return True
    except Exception as e:
        log.error(f"Failed to wake chat model {model_id}: {e}")

    return False


async def wake_embedding_model(model_id: str) -> bool:
    """Wake up an embedding model."""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model_id,
            "input": ["wake up"],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{API_ENDPOINT}/embeddings",
                    headers=headers,
                    json=data,
                    timeout=TASK_TIMEOUT
            ) as response:
                if response.status == 200:
                    model_last_activity[model_id] = time.time()
                    log.info(f"Embedding model {model_id} awakened")
                    return True
    except Exception as e:
        log.error(f"Failed to wake embedding model {model_id}: {e}")

    return False


async def wake_reranker_model(model_id: str) -> bool:
    """Wake up a reranker model."""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model_id,
            "query": "wake up",
            "documents": ["test document"],
            "top_k": 1
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{API_ENDPOINT}/rerank",
                    headers=headers,
                    json=data,
                    timeout=TASK_TIMEOUT
            ) as response:
                if response.status == 200:
                    model_last_activity[model_id] = time.time()
                    log.info(f"Reranker model {model_id} awakened")
                    return True
    except Exception as e:
        log.error(f"Failed to wake reranker model {model_id}: {e}")

    return False