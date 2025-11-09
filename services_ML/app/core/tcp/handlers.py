"""
Handlers TCP que conectan comandos con casos de uso.
"""
from typing import Dict, Any
from loguru import logger

from app.features.clustering.application.use_cases.cluster_documents import ClusterDocumentsUseCase
from app.features.clustering.application.use_cases.get_clusters import GetClustersUseCase
from app.features.topic_modeling.application.use_cases.extract_topics import ExtractTopicsUseCase
from app.features.recommendations.application.use_cases.recommend_similar import RecommendSimilarUseCase
from app.features.visualization.application.use_cases.update_visualization import UpdateVisualizationUseCase


# ========================================
# Clustering Handlers
# ========================================


async def handle_cluster_documents(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler para cluster_documents.

    Args:
        data: {'user_id': str, 'document_ids': list, 'force_recluster': bool}

    Returns:
        Resultado del clustering
    """
    logger.info(f"TCP Handler: cluster_documents para user_id={data.get('user_id')}")

    use_case = ClusterDocumentsUseCase()
    result = await use_case.execute(
        user_id=data.get("user_id"),
        document_ids=data.get("document_ids"),
        force_recluster=data.get("force_recluster", False),
    )

    return result


async def handle_get_clusters(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler para get_clusters.

    Args:
        data: {'user_id': str}

    Returns:
        Clusters del usuario
    """
    logger.info(f"TCP Handler: get_clusters para user_id={data.get('user_id')}")

    use_case = GetClustersUseCase()
    result = await use_case.execute(user_id=data.get("user_id"))

    return result


# ========================================
# Topic Modeling Handlers
# ========================================


async def handle_extract_topics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler para extract_topics.

    Args:
        data: {'user_id': str, 'num_topics': int, 'document_ids': list}

    Returns:
        Temas extraídos
    """
    logger.info(f"TCP Handler: extract_topics para user_id={data.get('user_id')}")

    use_case = ExtractTopicsUseCase()
    result = await use_case.execute(
        user_id=data.get("user_id"),
        num_topics=data.get("num_topics"),
        document_ids=data.get("document_ids"),
    )

    return result


# ========================================
# Recommendations Handlers
# ========================================


async def handle_recommend_similar(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler para recommend_similar.

    Args:
        data: {'document_id': str, 'top_k': int, 'user_id': str}

    Returns:
        Documentos similares
    """
    logger.info(f"TCP Handler: recommend_similar para document_id={data.get('document_id')}")

    use_case = RecommendSimilarUseCase()
    result = await use_case.execute(
        document_id=data.get("document_id"),
        top_k=data.get("top_k", 5),
        user_id=data.get("user_id"),
    )

    return result


# ========================================
# Visualization Handlers
# ========================================


async def handle_update_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler para update_visualization.

    Args:
        data: {'user_id': str, 'force_update': bool}

    Returns:
        Visualización 2D
    """
    logger.info(f"TCP Handler: update_visualization para user_id={data.get('user_id')}")

    use_case = UpdateVisualizationUseCase()
    result = await use_case.execute(
        user_id=data.get("user_id"), force_update=data.get("force_update", False)
    )

    return result


async def handle_get_visualization(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler para get_visualization.

    Args:
        data: {'user_id': str}

    Returns:
        Visualización existente
    """
    logger.info(f"TCP Handler: get_visualization para user_id={data.get('user_id')}")

    use_case = UpdateVisualizationUseCase()
    result = await use_case.execute(user_id=data.get("user_id"), force_update=False)

    return result


# ========================================
# Temporal Analysis Handlers (TODO)
# ========================================


async def handle_analyze_trends(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler para analyze_trends.

    Args:
        data: {'user_id': str, 'window_days': int}

    Returns:
        Tendencias temporales
    """
    logger.info(f"TCP Handler: analyze_trends para user_id={data.get('user_id')}")

    # TODO: Implementar caso de uso de análisis temporal
    return {
        "success": False,
        "error": "Análisis temporal no implementado aún",
    }


# ========================================
# Registry Function
# ========================================


def register_tcp_handlers(tcp_server):
    """
    Registra todos los handlers TCP en el servidor.

    Args:
        tcp_server: Instancia del servidor TCP
    """
    from app.core.tcp.protocol import TCPAction

    logger.info("Registrando handlers TCP...")

    # Clustering
    tcp_server.register_handler(TCPAction.CLUSTER_DOCUMENTS, handle_cluster_documents)
    tcp_server.register_handler(TCPAction.GET_CLUSTERS, handle_get_clusters)

    # Topics
    tcp_server.register_handler(TCPAction.EXTRACT_TOPICS, handle_extract_topics)

    # Recommendations
    tcp_server.register_handler(TCPAction.RECOMMEND_SIMILAR, handle_recommend_similar)

    # Visualization
    tcp_server.register_handler(TCPAction.UPDATE_VISUALIZATION, handle_update_visualization)
    tcp_server.register_handler(TCPAction.GET_VISUALIZATION, handle_get_visualization)

    # Temporal Analysis
    tcp_server.register_handler(TCPAction.ANALYZE_TRENDS, handle_analyze_trends)

    logger.info("✅ Handlers TCP registrados exitosamente")
