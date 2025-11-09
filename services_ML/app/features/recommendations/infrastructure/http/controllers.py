"""
Controllers REST para recomendaciones.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.features.recommendations.application.use_cases.recommend_similar import RecommendSimilarUseCase

router = APIRouter()


class RecommendationRequest(BaseModel):
    """Request para recomendaciones."""

    document_id: str = Field(..., description="ID del documento base")
    top_k: int = Field(default=5, description="Número de recomendaciones")
    user_id: Optional[str] = Field(default=None, description="Filtrar por usuario")


@router.post("/similar")
async def recommend_similar(request: RecommendationRequest):
    """
    Encuentra documentos similares.

    Usa similitud coseno en embeddings para encontrar documentos relacionados.
    """
    use_case = RecommendSimilarUseCase()
    result = await use_case.execute(
        document_id=request.document_id,
        top_k=request.top_k,
        user_id=request.user_id,
    )

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "No se encontraron recomendaciones"))

    return result
