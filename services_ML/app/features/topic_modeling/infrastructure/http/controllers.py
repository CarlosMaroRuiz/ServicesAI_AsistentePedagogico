"""
Controllers REST para topic modeling.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from app.features.topic_modeling.application.use_cases.extract_topics import ExtractTopicsUseCase

router = APIRouter()


class TopicsRequest(BaseModel):
    """Request para extracción de temas."""

    user_id: str = Field(..., description="ID del usuario")
    num_topics: Optional[int] = Field(default=None, description="Número de temas (auto si None)")
    document_ids: Optional[List[str]] = Field(default=None, description="IDs específicos (opcional)")


@router.post("/extract")
async def extract_topics(request: TopicsRequest):
    """
    Extrae temas principales de documentos.

    Usa BERTopic para identificar temas automáticamente.
    """
    use_case = ExtractTopicsUseCase()
    result = await use_case.execute(
        user_id=request.user_id,
        num_topics=request.num_topics,
        document_ids=request.document_ids,
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Error desconocido"))

    return result


@router.get("/{user_id}")
async def get_topics(user_id: str):
    """
    Obtiene los temas de un usuario.

    Retorna temas ya extraídos previamente.
    """
    from app.features.clustering.infrastructure.adapters import PersistenceAdapter

    persistence = PersistenceAdapter()

    with persistence.get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT * FROM v_topic_summary
                WHERE user_id = %s
                ORDER BY topic_id
            """
            cur.execute(query, (user_id,))
            topics = cur.fetchall()

    if not topics:
        raise HTTPException(status_code=404, detail="No se encontraron temas para este usuario")

    return {
        "success": True,
        "user_id": user_id,
        "total_topics": len(topics),
        "topics": [
            {
                "topic_id": t["topic_id"],
                "label": t["topic_label"],
                "keywords": t["keywords"],
                "document_count": t["actual_document_count"],
            }
            for t in topics
        ],
    }
