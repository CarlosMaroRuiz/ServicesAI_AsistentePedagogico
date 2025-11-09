"""
Controllers REST para clustering.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List

from app.features.clustering.application.use_cases.cluster_documents import ClusterDocumentsUseCase
from app.features.clustering.application.use_cases.get_clusters import GetClustersUseCase

router = APIRouter()


# ========================================
# Request/Response Models
# ========================================


class ClusterRequest(BaseModel):
    """Request para clustering."""

    user_id: str = Field(..., description="ID del usuario")
    document_ids: Optional[List[str]] = Field(default=None, description="IDs específicos (opcional)")
    force_recluster: bool = Field(default=False, description="Forzar re-clustering")


class ClusterResponse(BaseModel):
    """Response de clustering."""

    success: bool
    user_id: str | None = None
    total_documents: int | None = None
    num_clusters: int | None = None
    num_outliers: int | None = None
    outlier_percentage: float | None = None
    clusters: List[dict] | None = None
    error: str | None = None


# ========================================
# Endpoints
# ========================================


@router.post("/analyze", response_model=ClusterResponse)
async def analyze_documents(request: ClusterRequest):
    """
    Analiza y agrupa documentos automáticamente.

    Ejecuta clustering con HDBSCAN sobre embeddings reducidos con UMAP.
    """
    use_case = ClusterDocumentsUseCase()
    result = await use_case.execute(
        user_id=request.user_id,
        document_ids=request.document_ids,
        force_recluster=request.force_recluster,
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Error desconocido"))

    return result


@router.get("/clusters/{user_id}")
async def get_clusters(user_id: str):
    """
    Obtiene los clusters de un usuario.

    Retorna clusters ya calculados sin ejecutar clustering nuevamente.
    """
    use_case = GetClustersUseCase()
    result = await use_case.execute(user_id=user_id)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Clusters no encontrados"))

    return result


@router.get("/cluster/{cluster_id}")
async def get_cluster_details(cluster_id: int):
    """
    Obtiene detalles de un cluster específico.

    Incluye documentos, keywords, centroide, etc.
    """
    from app.features.clustering.infrastructure.adapters import PersistenceAdapter

    persistence = PersistenceAdapter()
    cluster_details = persistence.get_cluster_details(cluster_id)

    if not cluster_details:
        raise HTTPException(status_code=404, detail="Cluster no encontrado")

    return {
        "success": True,
        "cluster": cluster_details,
    }
