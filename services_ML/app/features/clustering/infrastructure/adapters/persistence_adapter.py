"""
Adaptador de persistencia para datos de clustering y ML.

Maneja todas las operaciones SQL para guardar y recuperar datos.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger
import psycopg

from app.core.database.connection import get_connection


class PersistenceAdapter:
    """Adaptador para operaciones de persistencia en PostgreSQL."""

    # ========================================
    # Embeddings Operations
    # ========================================

    @staticmethod
    def get_embeddings_by_user(user_id: str) -> Tuple[List[str], np.ndarray]:
        """
        Obtiene todos los embeddings de documentos de un usuario.

        Args:
            user_id: ID del usuario

        Returns:
            Tuple de (document_ids, embeddings_array)
        """
        logger.info(f"Obteniendo embeddings para user_id={user_id}")

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Obtener embeddings de la tabla langchain_pg_embedding
                query = """
                    SELECT DISTINCT ON (lpe.document)
                        lpe.document AS document_id,
                        lpe.embedding AS embedding
                    FROM langchain_pg_embedding lpe
                    JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
                    WHERE lpc.name = %s
                    AND lpe.document IS NOT NULL
                    ORDER BY lpe.document, lpe.id DESC
                """

                cur.execute(query, (user_id,))
                results = cur.fetchall()

                if not results:
                    logger.warning(f"No se encontraron embeddings para user_id={user_id}")
                    return [], np.array([])

                document_ids = [row["document_id"] for row in results]
                embeddings = np.array([row["embedding"] for row in results])

                logger.info(f"✅ Obtenidos {len(document_ids)} embeddings ({embeddings.shape})")

                return document_ids, embeddings

    @staticmethod
    def get_document_metadata(document_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene metadata de documentos.

        Args:
            document_ids: Lista de IDs de documentos

        Returns:
            Diccionario {document_id: {filename, pages, created_at, ...}}
        """
        if not document_ids:
            return {}

        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT id, filename, pages, file_size_mb, created_at
                    FROM documents
                    WHERE id = ANY(%s)
                """

                cur.execute(query, (document_ids,))
                results = cur.fetchall()

                metadata = {}
                for row in results:
                    metadata[row["id"]] = {
                        "filename": row["filename"],
                        "pages": row["pages"],
                        "file_size_mb": row["file_size_mb"],
                        "created_at": row["created_at"],
                    }

                return metadata

    # ========================================
    # Cluster Operations
    # ========================================

    @staticmethod
    def save_clusters(
        user_id: str,
        cluster_data: List[Dict[str, Any]],
        delete_existing: bool = True,
    ) -> None:
        """
        Guarda información de clusters en la base de datos.

        Args:
            user_id: ID del usuario
            cluster_data: Lista de diccionarios con datos de clusters
            delete_existing: Si eliminar clusters existentes del usuario
        """
        logger.info(f"Guardando {len(cluster_data)} clusters para user_id={user_id}")

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Eliminar clusters existentes si se solicita
                if delete_existing:
                    cur.execute("DELETE FROM ml_clusters WHERE user_id = %s", (user_id,))
                    logger.debug(f"Eliminados clusters existentes de {user_id}")

                # Insertar nuevos clusters
                for cluster in cluster_data:
                    query = """
                        INSERT INTO ml_clusters (
                            user_id, cluster_id, label, size, keywords, centroid
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, cluster_id)
                        DO UPDATE SET
                            label = EXCLUDED.label,
                            size = EXCLUDED.size,
                            keywords = EXCLUDED.keywords,
                            centroid = EXCLUDED.centroid,
                            updated_at = NOW()
                        RETURNING id
                    """

                    centroid = (
                        cluster["centroid"].tolist()
                        if isinstance(cluster["centroid"], np.ndarray)
                        else cluster["centroid"]
                    )

                    cur.execute(
                        query,
                        (
                            user_id,
                            cluster["cluster_id"],
                            cluster["label"],
                            cluster["size"],
                            cluster["keywords"],
                            centroid,
                        ),
                    )

                conn.commit()
                logger.info("✅ Clusters guardados exitosamente")

    @staticmethod
    def save_document_clusters(
        document_ids: List[str],
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
    ) -> None:
        """
        Guarda asignación de documentos a clusters.

        Args:
            document_ids: Lista de IDs de documentos
            labels: Array de etiquetas de cluster
            probabilities: Array de probabilidades (opcional)
        """
        logger.info(f"Guardando asignaciones de {len(document_ids)} documentos a clusters")

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Obtener IDs de clusters de la tabla ml_clusters
                cluster_id_map = {}
                cur.execute("SELECT id, cluster_id FROM ml_clusters")
                for row in cur.fetchall():
                    cluster_id_map[row["cluster_id"]] = row["id"]

                # Insertar asignaciones
                for i, doc_id in enumerate(document_ids):
                    cluster_label = int(labels[i])
                    is_outlier = cluster_label == -1
                    prob = float(probabilities[i]) if probabilities is not None else None

                    # Buscar el ID de la tabla ml_clusters
                    cluster_pk = cluster_id_map.get(cluster_label)

                    if cluster_pk is None and not is_outlier:
                        logger.warning(
                            f"Cluster {cluster_label} no encontrado en ml_clusters, saltando..."
                        )
                        continue

                    query = """
                        INSERT INTO ml_document_clusters (
                            document_id, cluster_id, probability, is_outlier
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (document_id) DO UPDATE SET
                            cluster_id = EXCLUDED.cluster_id,
                            probability = EXCLUDED.probability,
                            is_outlier = EXCLUDED.is_outlier
                    """

                    cur.execute(query, (doc_id, cluster_pk, prob, is_outlier))

                conn.commit()
                logger.info("✅ Asignaciones de documentos guardadas")

    @staticmethod
    def get_clusters_by_user(user_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene todos los clusters de un usuario.

        Args:
            user_id: ID del usuario

        Returns:
            Lista de clusters
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT * FROM v_cluster_summary
                    WHERE user_id = %s
                    ORDER BY cluster_id
                """

                cur.execute(query, (user_id,))
                return cur.fetchall()

    @staticmethod
    def get_document_cluster_labels(user_id: str, document_ids: List[str]) -> Dict[str, int]:
        """
        Obtiene el cluster_id (número) para cada documento.

        Args:
            user_id: ID del usuario
            document_ids: Lista de IDs de documentos

        Returns:
            Dict mapping document_id -> cluster_number (-1 si no tiene cluster)
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        dc.document_id::text,
                        c.cluster_id
                    FROM ml_document_clusters dc
                    JOIN ml_clusters c ON dc.cluster_id = c.id
                    WHERE c.user_id = %s
                      AND dc.document_id = ANY(%s)
                """

                cur.execute(query, (user_id, document_ids))
                results = cur.fetchall()

                # Crear dict con mapeo
                cluster_map = {}
                for row in results:
                    cluster_map[row["document_id"]] = row["cluster_id"]

                # Llenar con -1 los documentos sin cluster
                for doc_id in document_ids:
                    if doc_id not in cluster_map:
                        cluster_map[doc_id] = -1

                return cluster_map

    @staticmethod
    def get_cluster_details(cluster_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtiene detalles de un cluster específico.

        Args:
            cluster_id: ID del cluster

        Returns:
            Detalles del cluster
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        c.*,
                        array_agg(dc.document_id) AS document_ids
                    FROM ml_clusters c
                    LEFT JOIN ml_document_clusters dc ON c.id = dc.cluster_id
                    WHERE c.id = %s
                    GROUP BY c.id
                """

                cur.execute(query, (cluster_id,))
                return cur.fetchone()

    # ========================================
    # Topic Operations
    # ========================================

    @staticmethod
    def save_topics(
        user_id: str, topic_data: List[Dict[str, Any]], delete_existing: bool = True
    ) -> None:
        """
        Guarda temas extraídos en la base de datos.

        Args:
            user_id: ID del usuario
            topic_data: Lista de diccionarios con datos de temas
            delete_existing: Si eliminar temas existentes del usuario
        """
        logger.info(f"Guardando {len(topic_data)} temas para user_id={user_id}")

        with get_connection() as conn:
            with conn.cursor() as cur:
                if delete_existing:
                    cur.execute("DELETE FROM ml_topics WHERE user_id = %s", (user_id,))

                for topic in topic_data:
                    query = """
                        INSERT INTO ml_topics (
                            user_id, topic_id, label, keywords, document_count
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, topic_id)
                        DO UPDATE SET
                            label = EXCLUDED.label,
                            keywords = EXCLUDED.keywords,
                            document_count = EXCLUDED.document_count,
                            updated_at = NOW()
                    """

                    cur.execute(
                        query,
                        (
                            user_id,
                            topic["topic_id"],
                            topic["label"],
                            topic["keywords"],
                            topic["document_count"],
                        ),
                    )

                conn.commit()
                logger.info("✅ Temas guardados exitosamente")

    @staticmethod
    def save_document_topics(
        document_ids: List[str],
        topics: List[int],
        probabilities: Optional[np.ndarray] = None,
    ) -> None:
        """
        Guarda asignación de documentos a temas.

        Args:
            document_ids: Lista de IDs de documentos
            topics: Lista de IDs de temas
            probabilities: Array de probabilidades (opcional)
        """
        logger.info(f"Guardando asignaciones de {len(document_ids)} documentos a temas")

        with get_connection() as conn:
            with conn.cursor() as cur:
                # Obtener IDs de topics de la tabla ml_topics
                topic_id_map = {}
                cur.execute("SELECT id, topic_id FROM ml_topics")
                for row in cur.fetchall():
                    topic_id_map[row["topic_id"]] = row["id"]

                for i, doc_id in enumerate(document_ids):
                    topic_label = topics[i]
                    topic_pk = topic_id_map.get(topic_label)

                    if topic_pk is None:
                        continue

                    prob = (
                        float(probabilities[i][topic_label])
                        if probabilities is not None
                        else None
                    )

                    query = """
                        INSERT INTO ml_document_topics (
                            document_id, topic_id, probability
                        ) VALUES (%s, %s, %s)
                        ON CONFLICT (document_id, topic_id) DO UPDATE SET
                            probability = EXCLUDED.probability
                    """

                    cur.execute(query, (doc_id, topic_pk, prob))

                conn.commit()
                logger.info("✅ Asignaciones de temas guardadas")

    # ========================================
    # Visualization Operations
    # ========================================

    @staticmethod
    def save_visualization(
        user_id: str, document_ids: List[str], coordinates_2d: np.ndarray, labels: np.ndarray
    ) -> None:
        """
        Guarda coordenadas 2D para visualización.

        Args:
            user_id: ID del usuario
            document_ids: Lista de IDs de documentos
            coordinates_2d: Array de coordenadas (N x 2)
            labels: Array de etiquetas de cluster
        """
        logger.info(f"Guardando visualización 2D de {len(document_ids)} documentos")

        with get_connection() as conn:
            with conn.cursor() as cur:
                for i, doc_id in enumerate(document_ids):
                    x, y = float(coordinates_2d[i, 0]), float(coordinates_2d[i, 1])
                    cluster_id = int(labels[i])

                    query = """
                        INSERT INTO ml_visualizations (
                            user_id, document_id, x, y, cluster_id
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, document_id) DO UPDATE SET
                            x = EXCLUDED.x,
                            y = EXCLUDED.y,
                            cluster_id = EXCLUDED.cluster_id,
                            updated_at = NOW()
                    """

                    cur.execute(query, (user_id, doc_id, x, y, cluster_id))

                conn.commit()
                logger.info("✅ Visualización guardada")

    @staticmethod
    def get_visualization(user_id: str) -> List[Dict[str, Any]]:
        """
        Obtiene coordenadas de visualización de un usuario.

        Args:
            user_id: ID del usuario

        Returns:
            Lista de puntos 2D con metadata
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        v.document_id,
                        v.x,
                        v.y,
                        v.cluster_id,
                        c.label AS cluster_label,
                        d.filename
                    FROM ml_visualizations v
                    LEFT JOIN ml_clusters c ON v.cluster_id = c.cluster_id AND c.user_id = v.user_id
                    LEFT JOIN documents d ON v.document_id::uuid = d.id
                    WHERE v.user_id = %s
                """

                cur.execute(query, (user_id,))
                return cur.fetchall()

    # ========================================
    # Recommendations Operations
    # ========================================

    @staticmethod
    def save_recommendations(
        document_id: str, recommendations: List[Dict[str, Any]]
    ) -> None:
        """
        Guarda recomendaciones en cache.

        Args:
            document_id: ID del documento base
            recommendations: Lista de recomendaciones con similarity_score
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Eliminar recomendaciones existentes
                cur.execute(
                    "DELETE FROM ml_recommendations WHERE document_id = %s", (document_id,)
                )

                # Insertar nuevas
                for rank, rec in enumerate(recommendations, start=1):
                    query = """
                        INSERT INTO ml_recommendations (
                            document_id, recommended_document_id, similarity_score, rank
                        ) VALUES (%s, %s, %s, %s)
                    """

                    cur.execute(
                        query,
                        (
                            document_id,
                            rec["document_id"],
                            rec["similarity_score"],
                            rank,
                        ),
                    )

                conn.commit()

    @staticmethod
    def get_recommendations(document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Obtiene recomendaciones desde cache.

        Args:
            document_id: ID del documento
            top_k: Número de recomendaciones

        Returns:
            Lista de recomendaciones
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        r.recommended_document_id AS document_id,
                        r.similarity_score,
                        r.rank,
                        d.filename
                    FROM ml_recommendations r
                    LEFT JOIN documents d ON r.recommended_document_id::uuid = d.id
                    WHERE r.document_id = %s
                    ORDER BY r.rank
                    LIMIT %s
                """

                cur.execute(query, (document_id, top_k))
                return cur.fetchall()
