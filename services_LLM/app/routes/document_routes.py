"""
Rutas para gestión de documentos.
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.ingest_service import (
    process_and_store_pdf,
    get_document_by_id,
    list_user_documents
)
from app.services.update_service import (
    update_document,
    update_document_metadata,
    rename_document
)
from app.services.delete_service import (
    delete_document,
    delete_user_documents,
    restore_document
)
from app.services.pedagogical_service import (
    extract_pedagogical_content,
    save_pedagogical_content,
    get_pedagogical_content,
    search_pedagogical_content
)
from app.utils.text_extractor import extract_text_from_pdf
from app.models.document_model import (
    DocumentResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    PedagogicalContentResponse,
    PedagogicalSearchResponse
)
from app.models.response_model import UploadResponse, ErrorResponse

router = APIRouter(prefix="/documents", tags=["documents"])

# Configuración
UPLOAD_DIR = "temp"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Crear directorio de uploads si no existe
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    user_id: str = Query(..., min_length=1, max_length=100),
    file: UploadFile = File(...)
):
    """
    Sube un documento PDF y lo procesa.

    - **user_id**: ID del usuario que sube el documento
    - **file**: Archivo PDF a subir
    """
    try:
        # Validar tipo de archivo
        allowed_extensions = ['.pdf', '.txt']
        if not any(file.filename.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail="Solo se permiten archivos PDF o TXT"
            )

        # Validar tamaño
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Archivo muy grande. Máximo: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )

        # Sanitizar nombre de archivo
        safe_filename = Path(file.filename).name
        file_path = os.path.join(UPLOAD_DIR, f"{user_id}_{safe_filename}")

        # Guardar archivo
        with open(file_path, 'wb') as f:
            f.write(file_content)

        logger.info(f"Archivo guardado: {file_path}")

        # Procesar PDF
        result = process_and_store_pdf(
            user_id=user_id,
            file_path=file_path,
            filename=safe_filename
        )

        return UploadResponse(
            success=True,
            message="Documento subido y procesado exitosamente",
            document_id=result['document_id'],
            filename=safe_filename,
            chunks_created=result['chunks_created'],
            file_size_mb=result['file_size_mb']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al subir documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    user_id: str = Query(..., min_length=1),
    status: str = Query("active", pattern="^(active|inactive|deleted|all)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """
    Lista los documentos de un usuario.

    - **user_id**: ID del usuario
    - **status**: Filtro por estado (active, inactive, deleted, all)
    - **page**: Número de página
    - **page_size**: Documentos por página
    """
    try:
        if status == "all":
            status = None  # Sin filtro

        documents = list_user_documents(user_id, status)

        # Paginación simple
        total = len(documents)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_docs = documents[start:end]

        return DocumentListResponse(
            total=total,
            documents=paginated_docs,
            page=page,
            page_size=page_size
        )

    except Exception as e:
        logger.error(f"Error al listar documentos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    Obtiene información de un documento específico.

    - **document_id**: UUID del documento
    """
    try:
        document = get_document_by_id(document_id)

        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Documento no encontrado: {document_id}"
            )

        return document

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}", response_model=UploadResponse)
async def update_document_file(
    document_id: str,
    file: UploadFile = File(...)
):
    """
    Actualiza el archivo de un documento existente.

    - **document_id**: UUID del documento a actualizar
    - **file**: Nuevo archivo PDF
    """
    try:
        # Validar tipo
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Solo se permiten archivos PDF"
            )

        # Guardar archivo temporal
        file_content = await file.read()
        safe_filename = Path(file.filename).name
        temp_path = os.path.join(UPLOAD_DIR, f"update_{document_id}_{safe_filename}")

        with open(temp_path, 'wb') as f:
            f.write(file_content)

        # Actualizar documento
        result = update_document(
            document_id=document_id,
            new_file_path=temp_path,
            new_filename=safe_filename
        )

        return UploadResponse(
            success=True,
            message=f"Documento actualizado: {result['old_chunks']} -> {result['new_chunks']} chunks",
            document_id=document_id,
            filename=safe_filename,
            chunks_created=result['new_chunks'],
            file_size_mb=result['file_size_mb']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al actualizar documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{document_id}/rename")
async def rename_document_endpoint(
    document_id: str,
    new_filename: str = Query(..., min_length=1)
):
    """
    Renombra un documento.

    - **document_id**: UUID del documento
    - **new_filename**: Nuevo nombre
    """
    try:
        result = rename_document(document_id, new_filename)
        return result

    except Exception as e:
        logger.error(f"Error al renombrar documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document_endpoint(
    document_id: str,
    hard_delete: bool = Query(False, description="Eliminación permanente")
):
    """
    Elimina un documento.

    - **document_id**: UUID del documento
    - **hard_delete**: Si True, elimina permanentemente. Si False, marca como eliminado
    """
    try:
        result = delete_document(document_id, hard_delete)

        return DocumentDeleteResponse(
            success=True,
            message=result['message'],
            document_id=document_id,
            chunks_deleted=result['chunks_deleted']
        )

    except Exception as e:
        logger.error(f"Error al eliminar documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/user/{user_id}/all")
async def delete_all_user_documents(
    user_id: str,
    hard_delete: bool = Query(False)
):
    """
    Elimina todos los documentos de un usuario.

    - **user_id**: ID del usuario
    - **hard_delete**: Eliminación permanente
    """
    try:
        result = delete_user_documents(user_id, hard_delete)
        return result

    except Exception as e:
        logger.error(f"Error al eliminar documentos del usuario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/restore")
async def restore_document_endpoint(document_id: str):
    """
    Restaura un documento marcado como eliminado.

    - **document_id**: UUID del documento
    """
    try:
        result = restore_document(document_id)
        return result

    except Exception as e:
        logger.error(f"Error al restaurar documento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{document_id}/extract-pedagogical", response_model=PedagogicalContentResponse)
async def extract_pedagogical_endpoint(document_id: str):
    """
    Extrae contenido pedagógico de un documento usando DeepSeek API.

    Extrae:
    - Consejos pedagógicos
    - Ejercicios y actividades
    - Materiales didácticos
    - Objetivos de aprendizaje
    - Estrategias de enseñanza

    - **document_id**: UUID del documento
    """
    try:
        # Obtener documento
        document = get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Documento no encontrado")

        # Verificar que tenga file_path
        file_path = document.get('file_path')
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=400,
                detail="El documento no tiene archivo asociado o el archivo no existe"
            )

        logger.info(f"Extrayendo contenido pedagógico de documento {document_id}")

        # Extraer texto del PDF
        text = extract_text_from_pdf(file_path)

        # Extraer contenido pedagógico usando DeepSeek
        pedagogical_data = extract_pedagogical_content(
            text=text,
            filename=document['filename']
        )

        # Guardar en base de datos
        save_pedagogical_content(document_id, pedagogical_data)

        return PedagogicalContentResponse(
            success=True,
            document_id=document_id,
            filename=document['filename'],
            pedagogical_content=pedagogical_data,
            message="Contenido pedagógico extraído exitosamente"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al extraer contenido pedagógico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/pedagogical", response_model=PedagogicalContentResponse)
async def get_pedagogical_endpoint(document_id: str):
    """
    Obtiene el contenido pedagógico previamente extraído de un documento.

    - **document_id**: UUID del documento
    """
    try:
        # Obtener documento
        document = get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Documento no encontrado")

        # Obtener contenido pedagógico
        pedagogical_data = get_pedagogical_content(document_id)

        if not pedagogical_data:
            raise HTTPException(
                status_code=404,
                detail="No se ha extraído contenido pedagógico de este documento. Use POST /{document_id}/extract-pedagogical primero."
            )

        return PedagogicalContentResponse(
            success=True,
            document_id=document_id,
            filename=document['filename'],
            pedagogical_content=pedagogical_data,
            message="Contenido pedagógico recuperado exitosamente"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener contenido pedagógico: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pedagogical/search", response_model=PedagogicalSearchResponse)
async def search_pedagogical_endpoint(
    user_id: str = Query(..., min_length=1),
    query_type: str = Query(..., pattern="^(consejos|ejercicios|materiales|objetivos|estrategias)$"),
    keyword: Optional[str] = Query(None, min_length=1)
):
    """
    Busca contenido pedagógico específico en todos los documentos del usuario.

    - **user_id**: ID del usuario
    - **query_type**: Tipo de contenido (consejos, ejercicios, materiales, objetivos, estrategias)
    - **keyword**: Palabra clave opcional para filtrar resultados
    """
    try:
        results = search_pedagogical_content(
            user_id=user_id,
            query_type=query_type,
            keyword=keyword
        )

        return PedagogicalSearchResponse(
            success=True,
            query_type=query_type,
            total_results=len(results),
            results=results,
            message=f"Encontrados {len(results)} resultados para '{query_type}'"
        )

    except Exception as e:
        logger.error(f"Error al buscar contenido pedagógico: {e}")
        raise HTTPException(status_code=500, detail=str(e))
