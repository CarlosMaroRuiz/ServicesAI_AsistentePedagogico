"""
Caso de uso: Get Clusters.

Recupera clusters existentes de un usuario.
"""
from typing import Dict, Any, List
from loguru import logger

from app.features.clustering.infrastructure.adapters import PersistenceAdapter


class GetClustersUseCase:
    """Caso de uso para obtener clusters de un usuario."""

    def __init__(self):
        """Inicializa el caso de uso."""
        self.persistence = PersistenceAdapter()

    async def execute(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene los clusters de un usuario.

        Args:
            user_id: ID del usuario

        Returns:
            Diccionario con clusters
        """
        logger.info(f"Obteniendo clusters para user_id={user_id}")

        try:
            clusters = self.persistence.get_clusters_by_user(user_id)

            return {
                "success": True,
                "user_id": user_id,
                "total_clusters": len(clusters),
                "clusters": [
                    {
                        "cluster_id": c["cluster_id"],
                        "label": c["cluster_label"],
                        "size": c["actual_document_count"],
                        "keywords": c["keywords"],
                        "created_at": c["created_at"].isoformat() if c["created_at"] else None,
                    }
                    for c in clusters
                ],
            }

        except Exception as e:
            logger.error(f"Error obteniendo clusters: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
