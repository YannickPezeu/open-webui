# open_webui/routers/libraries.py
import os
import logging
import httpx
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from open_webui.models.users import Users
from open_webui.models.groups import Groups  # ✅ Importer Groups, pas UserGroup
from open_webui.utils.auth import get_current_user, get_admin_user
from open_webui.internal.db import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
LIBRARY_API_URL = os.getenv("LIBRARY_API_URL", "http://hierarchical-search-service:80")
LIBRARY_API_KEY = os.getenv("VITE_LIBRARY_API_KEY")

print(f"Library API URL: {LIBRARY_API_URL}")

if not LIBRARY_API_KEY:
    logger.error("⚠️ LIBRARY_API_KEY not set! Library features will not work.")


def get_user_group_names(user_id: str) -> List[str]:
    """
    Récupère la liste des group IDs d'un utilisateur.
    Utilise la méthode get_groups_by_member_id de GroupTable.
    """
    user_groups = Groups.get_groups_by_member_id(user_id)
    logger.info("user_groups: %s", user_groups)
    group_names = [group.name for group in user_groups]
    logger.debug(f"User {user_id} belongs to groups: {group_names}")
    return group_names


@router.post("/{library_id}/search")
async def search_library(
    library_id: str,
    query: str = Form(...),
    password: Optional[str] = Form(None),
    user: Users = Depends(get_current_user)
):
    """
    Proxy vers le FastAPI de recherche.
    Vérifie l'utilisateur et transmet ses groupes vérifiés.
    """
    logger.info(f"🔍 User {user.email} searching in library {library_id}")
    
    # Récupérer les groupes réels de l'utilisateur
    user_group_names = get_user_group_names(user.id)
    logger.info(f"User groups: {user_group_names}")
    
    if not user_group_names:
        logger.warning(f"User {user.email} has no groups assigned")
    
    # Appeler le FastAPI avec l'API key interne
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LIBRARY_API_URL}/search/{library_id}",
                headers={
                    "X-API-Key": LIBRARY_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "user_groups": user_group_names,
                    "password": password
                }
            )
            
            if response.status_code != 200:
                error_detail = response.json().get("detail", "Unknown error")
                logger.error(f"Search failed: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_detail
                )
            
            results = response.json()
            logger.info(f"✅ Search successful: {len(results)} results")
            return results
            
    except httpx.TimeoutException:
        logger.error(f"Timeout calling library API for {library_id}")
        raise HTTPException(
            status_code=504,
            detail="Search request timed out"
        )
    except httpx.RequestError as e:
        logger.error(f"Error calling library API: {e}")
        raise HTTPException(
            status_code=503,
            detail="Library search service unavailable"
        )


@router.post("/create")
async def create_library(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    groups: str = Form(...),  # JSON string des group IDs
    password: Optional[str] = Form(None),
    metadata_json: Optional[str] = Form(None),
    user: Users = Depends(get_current_user)
):
    """
    Crée une nouvelle library en appelant le FastAPI d'indexation.
    """
    import json
    import uuid
    
    logger.info(f"📚 User {user.email} creating library '{name}'")
    
    # Générer un ID unique pour la library
    library_id = str(uuid.uuid4())
    
    # Préparer les fichiers pour l'upload
    files_to_upload = []
    try:
        for file in files:
            file_content = await file.read()
            files_to_upload.append(
                ('files', (file.filename, file_content, file.content_type))
            )
            await file.seek(0)
        
        # Appeler le FastAPI pour créer l'index
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LIBRARY_API_URL}/index/{library_id}",
                headers={"X-API-Key": LIBRARY_API_KEY},
                files=files_to_upload,
                data={
                    "groups": groups,
                    "password": password,
                    "metadata_json": metadata_json
                }
            )
            
            if response.status_code != 202:
                error_detail = response.json().get("detail", "Unknown error")
                logger.error(f"Index creation failed: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to create index: {error_detail}"
                )
        
        # TODO: Enregistrer dans la DB Open WebUI si tu créées la table Library
        # Pour l'instant, on skippe cette partie
        
        logger.info(f"✅ Library {library_id} created successfully")
        return {
            "status": "success",
            "library_id": library_id,
            "message": "Library creation started. Indexing in progress."
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid groups JSON format")
    except Exception as e:
        logger.error(f"Error creating library: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_libraries(user: Users = Depends(get_current_user)):
    """
    Récupère toutes les libraries accessibles par l'utilisateur.
    Pour l'instant, retourne une liste vide (à implémenter avec la table Library).
    """
    # TODO: Implémenter quand la table Library sera créée
    return []


@router.delete("/{library_id}")
async def delete_library(
    library_id: str,
    user: Users = Depends(get_admin_user)
):
    """
    Supprime une library (admin uniquement).
    """
    # TODO: Implémenter quand la table Library sera créée
    logger.info(f"🗑️ Library {library_id} deletion requested by {user.email}")
    return {"status": "success", "message": "Not implemented yet"}