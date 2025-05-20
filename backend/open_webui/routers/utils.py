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

############ My code

import json
import time
import os
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter

router = APIRouter()

api_key = os.getenv("OPENAI_API_KEY")
API_ENDPOINT = os.getenv("OPENAI_API_BASE_URL", "https://inference-dev.rcp.epfl.ch/v1")

# Last wake-up timestamp tracker
last_wakeup_time = 0  # Unix timestamp of the last successful wake-up
WAKEUP_INTERVAL = 15 * 60  # 15 minutes in seconds

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


@router.get("/wake_up_models")
async def wake_up_models(force: bool = False):
    """
    Wake up models by concurrently sending requests to embedding and chat completion endpoints.
    Only wakes up models if it's been more than 15 minutes since the last wake-up or if force=True.
    """
    global last_wakeup_time
    current_time = time.time()

    # Calculate time elapsed since last wake-up
    elapsed_time = current_time - last_wakeup_time

    # Check if we need to wake up models
    if not force and last_wakeup_time > 0 and elapsed_time < WAKEUP_INTERVAL:
        minutes_ago = int(elapsed_time / 60)
        return {
            "status": "Models already awake",
            "last_wakeup": f"{minutes_ago} minutes ago",
            "next_wakeup_in": f"{int((WAKEUP_INTERVAL - elapsed_time) / 60)} minutes"
        }

    log.info("Starting model wake-up process...")
    try:
        # Run both API calls concurrently
        embedding_task, chat_task = await asyncio.gather(
            get_embeddings_async('a'),
            test_chat_completion_async(),
            return_exceptions=True  # This prevents one failed task from affecting the other
        )

        # Check results
        embedding_success = not isinstance(embedding_task, Exception) and embedding_task and embedding_task.get("data")
        chat_success = not isinstance(chat_task, Exception) and chat_task

        # Update last wake-up time if at least one model was successfully awakened
        if embedding_success or chat_success:
            last_wakeup_time = current_time
            log.info(
                f"Updated last_wakeup_time to {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}")

        if embedding_success and chat_success:
            log.info("Successfully woke up both embedding and chat models")
            return {"status": "All models are awake"}
        elif embedding_success:
            log.info("Successfully woke up embedding model, but chat model failed")
            return {"status": "Embedding model is awake, but chat model failed to wake up"}
        elif chat_success:
            log.info("Successfully woke up chat model, but embedding model failed")
            return {"status": "Chat model is awake, but embedding model failed to wake up"}
        else:
            log.warning("Failed to wake up both embedding and chat models")
            return {"status": "Failed to wake up models"}

    except Exception as e:
        log.error(f"Error waking up models: {str(e)}")
        return {"status": "Error", "message": str(e)}


# Rest of your code (get_embeddings_async, test_chat_completion_async, etc.) remains the same


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

        except aiohttp.ClientError as e:
            print(f"Embedding error: {e}")
            return None


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

        except aiohttp.ClientError as e:
            print(f"Chat completion error: {e}")
            return False


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
