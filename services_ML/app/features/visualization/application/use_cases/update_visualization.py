"""
Caso de uso: Update Visualization.

Genera coordenadas 2D para visualización de documentos.
"""
import numpy as np
from typing import Dict, Any, Optional, List
from loguru import logger

from app.features.clustering.infrastructure.adapters import UMAPAdapter, PersistenceAdapter


class UpdateVisualizationUseCase:
    """Caso de uso para actualizar visualización 2D de documentos."""

    def __init__(self):
        """Inicializa el caso de uso."""
        self.persistence = PersistenceAdapter()

    async def execute(self, user_id: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Genera visualización 2D de documentos.

        Args:
            user_id: ID del usuario
            force_update: Forzar actualización

        Returns:
            Diccionario con puntos 2D
        """
        logger.info(f"=== Generando visualización 2D para user_id={user_id} ===")

        try:
            # Si no es forzado, intentar obtener visualización existente
            if not force_update:
                existing_viz = self.persistence.get_visualization(user_id)
                if existing_viz:
                    logger.info(f"Retornando visualización existente ({len(existing_viz)} puntos)")
                    return self._format_response(user_id, existing_viz)

            # 1. Obtener embeddings
            doc_ids, embeddings = self.persistence.get_embeddings_by_user(user_id)

            if len(doc_ids) == 0:
                return {
                    "success": False,
                    "error": f"No se encontraron documentos para user_id={user_id}",
                }

            # 2. Reducir a 2D con UMAP
            umap_adapter = UMAPAdapter.create_for_visualization()
            coordinates_2d = umap_adapter.fit_transform(embeddings)

            logger.info(f"Reducción 2D completada: {embeddings.shape} → {coordinates_2d.shape}")

            # 3. Obtener labels de clusters (si existen)
            clusters = self.persistence.get_clusters_by_user(user_id)
            if clusters:
                # Obtener assignments de clusters por documento
                cluster_map = self.persistence.get_document_cluster_labels(user_id, doc_ids)
                # Convertir a array numpy manteniendo el orden de doc_ids
                labels = np.array([cluster_map.get(doc_id, -1) for doc_id in doc_ids], dtype=int)
            else:
                labels = np.full(len(doc_ids), -1)

            # 4. Guardar visualización
            self.persistence.save_visualization(user_id, doc_ids, coordinates_2d, labels)

            # 5. Obtener visualización guardada con metadata
            viz_data = self.persistence.get_visualization(user_id)

            logger.info("=== Visualización generada exitosamente ===")

            return self._format_response(user_id, viz_data)

        except Exception as e:
            logger.error(f"Error generando visualización: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _format_response(self, user_id: str, viz_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formatea la respuesta de visualización.

        Args:
            user_id: ID del usuario
            viz_data: Datos de visualización

        Returns:
            Respuesta formateada
        """
        points = [
            {
                "document_id": v["document_id"],
                "x": float(v["x"]),
                "y": float(v["y"]),
                "cluster_id": v.get("cluster_id", -1),
                "cluster_label": v.get("cluster_label", "Sin cluster"),
                "filename": v.get("filename", ""),
            }
            for v in viz_data
        ]

        return {
            "success": True,
            "user_id": user_id,
            "total_points": len(points),
            "points": points,
        }
