"""
Controllers REST para visualización.
"""
from fastapi import APIRouter, HTTPException, Query

from app.features.visualization.application.use_cases.update_visualization import UpdateVisualizationUseCase

router = APIRouter()


@router.post("/update/{user_id}")
async def update_visualization(user_id: str, force_update: bool = Query(default=False)):
    """
    Actualiza visualización 2D de documentos.

    Genera coordenadas 2D usando UMAP para visualización interactiva.
    """
    use_case = UpdateVisualizationUseCase()
    result = await use_case.execute(user_id=user_id, force_update=force_update)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Error generando visualización"))

    return result


@router.get("/clusters-2d/{user_id}")
async def get_visualization(user_id: str):
    """
    Obtiene coordenadas 2D para visualización de clusters.

    Retorna visualización ya generada previamente.
    """
    use_case = UpdateVisualizationUseCase()
    result = await use_case.execute(user_id=user_id, force_update=False)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Visualización no encontrada"))

    return result
