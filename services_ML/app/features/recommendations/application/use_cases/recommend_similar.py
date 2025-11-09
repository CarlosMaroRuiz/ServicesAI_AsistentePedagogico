"""
Caso de uso: Recommend Similar.

Encuentra documentos similares usando embeddings.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from app.features.clustering.infrastructure.adapters import PersistenceAdapter
from app.core.database.connection import get_connection


class RecommendSimilarUseCase:
    """Caso de uso para recomendar documentos similares."""

    def __init__(self):
        """Inicializa el caso de uso."""
        self.persistence = PersistenceAdapter()

    async def execute(
        self, document_id: str, top_k: int = 5, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encuentra documentos similares.

        Args:
            document_id: ID del documento base
            top_k: Número de recomendaciones
            user_id: Filtrar por usuario (opcional)

        Returns:
            Diccionario con recomendaciones
        """
        logger.info(f"Buscando {top_k} documentos similares a {document_id}")

        try:
            # Intentar obtener desde cache
            cached_recs = self.persistence.get_recommendations(document_id, top_k)
            if cached_recs:
                logger.info(f"Retornando {len(cached_recs)} recomendaciones desde cache")
                return {
                    "success": True,
                    "document_id": document_id,
                    "recommendations": [
                        {
                            "document_id": r["document_id"],
                            "filename": r["filename"],
                            "similarity_score": round(r["similarity_score"], 4),
                            "rank": r["rank"],
                        }
                        for r in cached_recs
                    ],
                }

            # Si no hay cache, calcular similitud usando pgvector
            recommendations = self._calculate_similarity(document_id, top_k, user_id)

            if not recommendations:
                return {
                    "success": False,
                    "error": "No se encontraron documentos similares",
                }

            # Guardar en cache
            self.persistence.save_recommendations(document_id, recommendations)

            logger.info(f"Calculadas y guardadas {len(recommendations)} recomendaciones")

            return {
                "success": True,
                "document_id": document_id,
                "recommendations": [
                    {
                        "document_id": r["document_id"],
                        "filename": r["filename"],
                        "similarity_score": round(r["similarity_score"], 4),
                    }
                    for r in recommendations
                ],
            }

        except Exception as e:
            logger.error(f"Error calculando recomendaciones: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _calculate_similarity(
        self, document_id: str, top_k: int, user_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Calcula similitud usando pgvector.

        Args:
            document_id: ID del documento base
            top_k: Número de recomendaciones
            user_id: Filtrar por usuario

        Returns:
            Lista de documentos similares
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Buscar embedding del documento base
                query_base = """
                    SELECT embedding AS embedding, collection_id
                    FROM langchain_pg_embedding
                    WHERE document = %s
                    LIMIT 1
                """
                cur.execute(query_base, (document_id,))
                base_result = cur.fetchone()

                if not base_result:
                    return []

                base_embedding = base_result["embedding"]
                collection_id = base_result["collection_id"]

                # Buscar documentos similares usando similitud coseno
                query_similar = """
                    SELECT DISTINCT ON (lpe.document)
                        lpe.document AS document_id,
                        d.filename,
                        1 - (lpe.embedding <=> %s::vector) AS similarity_score
                    FROM langchain_pg_embedding lpe
                    LEFT JOIN documents d ON lpe.document::uuid = d.id
                    WHERE lpe.collection_id = %s
                    AND lpe.document != %s
                    AND lpe.document IS NOT NULL
                    ORDER BY lpe.document, lpe.embedding <=> %s::vector
                    LIMIT %s
                """

                cur.execute(
                    query_similar,
                    (base_embedding, collection_id, document_id, base_embedding, top_k),
                )
                results = cur.fetchall()

                return [
                    {
                        "document_id": r["document_id"],
                        "filename": r["filename"] or "Sin nombre",
                        "similarity_score": float(r["similarity_score"]),
                    }
                    for r in results
                ]
