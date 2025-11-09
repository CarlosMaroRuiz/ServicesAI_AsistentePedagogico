"""
Rutas para funcionalidades de Machine Learning.

Conecta con services_ML vía TCP para clustering, topic modeling,
recomendaciones y visualización.
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from app.shared.ml_client.tcp_client import create_ml_client


router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# ========================================
# Request Models
# ========================================

class ClusterRequest(BaseModel):
    """Request para clustering de documentos."""
    user_id: str
    document_ids: Optional[List[str]] = None
    force_recluster: bool = False


class TopicRequest(BaseModel):
    """Request para extracción de temas."""
    user_id: str
    num_topics: Optional[int] = None
    document_ids: Optional[List[str]] = None


class RecommendationRequest(BaseModel):
    """Request para recomendaciones."""
    document_id: str
    top_k: int = 5
    user_id: Optional[str] = None


class VisualizationRequest(BaseModel):
    """Request para actualización de visualización."""
    user_id: str
    force_update: bool = False


class TrendsRequest(BaseModel):
    """Request para análisis de tendencias."""
    user_id: str
    window_days: int = 30


# ========================================
# Endpoints - Clustering
# ========================================

@router.post("/cluster")
async def cluster_documents(request: ClusterRequest):
    """
    Agrupa documentos del usuario usando HDBSCAN.

    - Reduce embeddings de 384D a 5D con UMAP
    - Aplica clustering HDBSCAN
    - Genera etiquetas descriptivas para cada cluster
    - Guarda resultados en la base de datos ML

    Returns:
        Información de clusters encontrados
    """
    try:
        logger.info(f"Solicitando clustering para user_id={request.user_id}")

        ml_client = create_ml_client()
        result = await ml_client.cluster_documents(
            user_id=request.user_id,
            document_ids=request.document_ids,
            force_recluster=request.force_recluster
        )

        logger.info(f"Clustering completado: {result.get('num_clusters', 0)} clusters")
        return {
            "success": True,
            "data": result
        }

    except ConnectionError as e:
        logger.error(f"Error de conexión con ML server: {e}")
        raise HTTPException(
            status_code=503,
            detail="Servicio ML no disponible. Asegúrate de que services_ML esté ejecutándose."
        )
    except TimeoutError as e:
        logger.error(f"Timeout en clustering: {e}")
        raise HTTPException(status_code=504, detail="Timeout en operación de clustering")
    except Exception as e:
        logger.error(f"Error en clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{user_id}")
async def get_clusters(user_id: str):
    """
    Obtiene todos los clusters de un usuario.

    Returns:
        Lista de clusters con sus documentos
    """
    try:
        ml_client = create_ml_client()
        result = await ml_client.send_request("get_clusters", {"user_id": user_id})

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error obteniendo clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clusters/{user_id}/{cluster_id}/documents")
async def get_cluster_documents(user_id: str, cluster_id: str):
    """
    Obtiene los documentos de un cluster específico.

    Returns:
        Lista de documentos en el cluster
    """
    try:
        ml_client = create_ml_client()
        result = await ml_client.send_request(
            "get_cluster_documents",
            {"user_id": user_id, "cluster_id": cluster_id}
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error obteniendo documentos del cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# Endpoints - Topic Modeling
# ========================================

@router.post("/topics")
async def extract_topics(request: TopicRequest):
    """
    Extrae temas de los documentos usando BERTopic.

    - Usa embeddings pre-calculados
    - Aplica UMAP + HDBSCAN + c-TF-IDF
    - Genera etiquetas descriptivas para cada tema
    - Guarda resultados en la base de datos ML

    Returns:
        Temas extraídos con keywords y documentos asociados
    """
    try:
        logger.info(f"Solicitando extracción de temas para user_id={request.user_id}")

        ml_client = create_ml_client()
        result = await ml_client.extract_topics(
            user_id=request.user_id,
            num_topics=request.num_topics,
            document_ids=request.document_ids
        )

        logger.info(f"Extracción completada: {result.get('num_topics', 0)} temas")
        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error en extracción de temas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topics/{user_id}")
async def get_topics(user_id: str):
    """
    Obtiene todos los temas de un usuario.

    Returns:
        Lista de temas con keywords y documentos
    """
    try:
        ml_client = create_ml_client()
        result = await ml_client.send_request("get_topics", {"user_id": user_id})

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error obteniendo temas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# Endpoints - Recommendations
# ========================================

@router.post("/recommend")
async def recommend_similar(request: RecommendationRequest):
    """
    Recomienda documentos similares basándose en embeddings.

    - Usa búsqueda KNN en espacio vectorial
    - Calcula similitud coseno
    - Opcionalmente filtra por usuario

    Returns:
        Lista de documentos similares ordenados por relevancia
    """
    try:
        logger.info(f"Solicitando recomendaciones para doc_id={request.document_id}")

        ml_client = create_ml_client()
        result = await ml_client.recommend_similar(
            document_id=request.document_id,
            top_k=request.top_k,
            user_id=request.user_id
        )

        logger.info(f"Recomendaciones generadas: {len(result.get('recommendations', []))} docs")
        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error en recomendaciones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# Endpoints - Visualization
# ========================================

@router.post("/visualization")
async def update_visualization(request: VisualizationRequest):
    """
    Genera datos de visualización 2D para documentos.

    - Reduce embeddings de 384D a 2D con UMAP
    - Incluye etiquetas de clusters y topics
    - Optimizado para ploteo en frontend

    Returns:
        Coordenadas 2D con metadatos para visualización
    """
    try:
        logger.info(f"Solicitando visualización para user_id={request.user_id}")

        ml_client = create_ml_client()
        result = await ml_client.update_visualization(
            user_id=request.user_id,
            force_update=request.force_update
        )

        logger.info(f"Visualización generada: {result.get('num_points', 0)} puntos")
        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error en visualización: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================
# Endpoints - Health & Status
# ========================================

@router.get("/health")
async def ml_health_check():
    """
    Verifica la conexión con el servidor ML.

    Returns:
        Estado del servidor ML
    """
    try:
        ml_client = create_ml_client(timeout=5)
        result = await ml_client.ping()

        return {
            "success": True,
            "status": "healthy",
            "ml_server": "available",
            "data": result
        }

    except ConnectionError:
        return {
            "success": False,
            "status": "unhealthy",
            "ml_server": "unavailable",
            "message": "No se puede conectar a services_ML en localhost:5555"
        }
    except Exception as e:
        return {
            "success": False,
            "status": "error",
            "message": str(e)
        }


@router.get("/status")
async def ml_status():
    """
    Obtiene información de estado del servidor ML.

    Returns:
        Estadísticas y estado del servidor ML
    """
    try:
        ml_client = create_ml_client()
        result = await ml_client.get_status()

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error obteniendo estado ML: {e}")
        raise HTTPException(status_code=500, detail=str(e))
