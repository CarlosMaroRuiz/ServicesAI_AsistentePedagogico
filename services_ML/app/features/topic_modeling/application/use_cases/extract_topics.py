"""
Caso de uso: Extract Topics.

Extrae temas principales usando BERTopic.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from app.features.topic_modeling.infrastructure.adapters import BERTopicAdapter
from app.features.clustering.infrastructure.adapters import PersistenceAdapter, UMAPAdapter, HDBSCANAdapter


class ExtractTopicsUseCase:
    """Caso de uso para extraer temas de documentos."""

    def __init__(self):
        """Inicializa el caso de uso."""
        self.persistence = PersistenceAdapter()

    async def execute(
        self,
        user_id: str,
        num_topics: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extrae temas de documentos.

        Args:
            user_id: ID del usuario
            num_topics: Número de temas (None = automático)
            document_ids: IDs específicos (opcional)

        Returns:
            Diccionario con temas extraídos
        """
        logger.info(f"=== Extrayendo temas para user_id={user_id} ===")

        try:
            # 1. Obtener embeddings y documentos
            doc_ids, embeddings = self.persistence.get_embeddings_by_user(user_id)

            if len(doc_ids) == 0:
                return {
                    "success": False,
                    "error": f"No se encontraron documentos para user_id={user_id}",
                }

            # Obtener textos de documentos (desde metadata/chunks)
            # Por simplicidad, usamos filenames como "documentos"
            # En producción, obtener textos completos
            metadata = self.persistence.get_document_metadata(doc_ids)
            documents = [metadata.get(doc_id, {}).get("filename", f"Doc {doc_id}") for doc_id in doc_ids]

            logger.info(f"Procesando {len(documents)} documentos para topic modeling")

            # 2. Preparar modelos (UMAP + HDBSCAN para BERTopic)
            umap_model = UMAPAdapter.create_for_clustering().reducer
            hdbscan_model = HDBSCANAdapter().clusterer

            # 3. Ejecutar BERTopic
            bertopic_adapter = BERTopicAdapter(nr_topics=num_topics or "auto")
            topics, probabilities = bertopic_adapter.fit(
                documents, embeddings, umap_reducer=umap_model, hdbscan_clusterer=hdbscan_model
            )

            # 4. Obtener información de temas
            topic_info = bertopic_adapter.get_topic_info()
            stats = bertopic_adapter.get_topic_statistics()

            # 5. Guardar en base de datos
            self.persistence.save_topics(user_id, topic_info, delete_existing=True)
            self.persistence.save_document_topics(doc_ids, topics, probabilities)

            logger.info("=== Extracción de temas completada ===")

            # 6. Retornar resultado
            return {
                "success": True,
                "user_id": user_id,
                "total_documents": len(documents),
                "num_topics": stats["num_topics"],
                "num_outliers": stats["num_outliers"],
                "topics": [
                    {
                        "topic_id": t["topic_id"],
                        "label": t["label"],
                        "keywords": t["keywords"],
                        "document_count": t["document_count"],
                    }
                    for t in topic_info
                ],
            }

        except Exception as e:
            logger.error(f"Error extrayendo temas: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
