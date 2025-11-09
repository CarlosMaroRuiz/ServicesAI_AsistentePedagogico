"""
Caso de uso: Cluster Documents.

Ejecuta clustering automático de documentos usando HDBSCAN + UMAP.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger

from app.features.clustering.infrastructure.adapters import (
    HDBSCANAdapter,
    UMAPAdapter,
    PersistenceAdapter,
)


class ClusterDocumentsUseCase:
    """Caso de uso para agrupar documentos automáticamente."""

    def __init__(self):
        """Inicializa el caso de uso."""
        self.persistence = PersistenceAdapter()

    async def execute(
        self, user_id: str, document_ids: Optional[List[str]] = None, force_recluster: bool = False
    ) -> Dict[str, Any]:
        """
        Ejecuta clustering de documentos.

        Args:
            user_id: ID del usuario
            document_ids: IDs específicos (opcional, None = todos los documentos)
            force_recluster: Forzar re-clustering aunque ya exista

        Returns:
            Diccionario con resultados del clustering
        """
        logger.info(f"=== Iniciando clustering para user_id={user_id} ===")

        try:
            # 1. Obtener embeddings
            doc_ids, embeddings = self.persistence.get_embeddings_by_user(user_id)

            if len(doc_ids) == 0:
                return {
                    "success": False,
                    "error": f"No se encontraron documentos para user_id={user_id}",
                }

            if len(doc_ids) < 3:
                return {
                    "success": False,
                    "error": f"Se necesitan al menos 3 documentos para clustering (encontrados: {len(doc_ids)})",
                }

            # Filtrar por document_ids si se especificó
            if document_ids:
                indices = [i for i, doc_id in enumerate(doc_ids) if doc_id in document_ids]
                doc_ids = [doc_ids[i] for i in indices]
                embeddings = embeddings[indices]

            logger.info(f"Procesando {len(doc_ids)} documentos con embeddings de {embeddings.shape[1]}D")

            # 2. Reducción dimensional con UMAP (384D → 5D para clustering)
            umap_adapter = UMAPAdapter.create_for_clustering()
            reduced_embeddings = umap_adapter.fit_transform(embeddings)

            # 3. Clustering con HDBSCAN
            hdbscan_adapter = HDBSCANAdapter()
            labels, probabilities = hdbscan_adapter.fit(reduced_embeddings)

            # 4. Obtener estadísticas
            stats = hdbscan_adapter.get_cluster_statistics()
            centroids = hdbscan_adapter.get_cluster_centroids(reduced_embeddings)

            # 5. Generar etiquetas descriptivas
            metadata = self.persistence.get_document_metadata(doc_ids)
            cluster_labels = self._generate_cluster_labels(doc_ids, labels, metadata)

            # 6. Preparar datos para guardar
            cluster_data = []
            for cluster_id in stats["cluster_ids"]:
                cluster_data.append(
                    {
                        "cluster_id": cluster_id,
                        "label": cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                        "size": stats["cluster_sizes"][cluster_id],
                        "keywords": self._extract_keywords_from_filenames(
                            doc_ids, labels, cluster_id, metadata
                        ),
                        "centroid": centroids[cluster_id],
                    }
                )

            # 7. Guardar en base de datos
            self.persistence.save_clusters(user_id, cluster_data, delete_existing=force_recluster)
            self.persistence.save_document_clusters(doc_ids, labels, probabilities)

            logger.info("=== Clustering completado exitosamente ===")

            # 8. Retornar resultado
            return {
                "success": True,
                "user_id": user_id,
                "total_documents": len(doc_ids),
                "num_clusters": stats["num_clusters"],
                "num_outliers": stats["num_outliers"],
                "outlier_percentage": round(stats["outlier_percentage"], 2),
                "clusters": [
                    {
                        "cluster_id": c["cluster_id"],
                        "label": c["label"],
                        "size": c["size"],
                        "keywords": c["keywords"],
                    }
                    for c in cluster_data
                ],
            }

        except Exception as e:
            logger.error(f"Error en clustering: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _generate_cluster_labels(
        self, doc_ids: List[str], labels: np.ndarray, metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[int, str]:
        """
        Genera etiquetas descriptivas para clusters basadas en nombres de archivos.

        Args:
            doc_ids: Lista de IDs de documentos
            labels: Array de etiquetas de cluster
            metadata: Metadata de documentos

        Returns:
            Diccionario {cluster_id: label}
        """
        cluster_labels = {}
        unique_labels = np.unique(labels)
        clusters = unique_labels[unique_labels >= 0]

        for cluster_id in clusters:
            # Obtener filenames del cluster
            cluster_doc_ids = [doc_ids[i] for i, label in enumerate(labels) if label == cluster_id]
            filenames = [
                metadata.get(doc_id, {}).get("filename", "")
                for doc_id in cluster_doc_ids
                if doc_id in metadata
            ]

            # Generar etiqueta simple (pueden mejorarse con NLP)
            if filenames:
                # Tomar palabras comunes de los nombres de archivo
                common_words = self._find_common_words(filenames)
                if common_words:
                    label = f"{common_words[0].title()}"
                else:
                    label = f"Documentos - Grupo {cluster_id}"
            else:
                label = f"Cluster {cluster_id}"

            cluster_labels[int(cluster_id)] = label

        return cluster_labels

    def _find_common_words(self, filenames: List[str], min_length: int = 4) -> List[str]:
        """
        Encuentra palabras comunes en nombres de archivos.

        Args:
            filenames: Lista de nombres de archivo
            min_length: Longitud mínima de palabra

        Returns:
            Lista de palabras comunes
        """
        from collections import Counter
        import re

        # Extraer palabras de filenames
        all_words = []
        for filename in filenames:
            # Remover extensión
            name = filename.rsplit(".", 1)[0]
            # Separar por guiones bajos, guiones y espacios
            words = re.split(r"[-_\s]+", name.lower())
            # Filtrar palabras cortas y números
            words = [w for w in words if len(w) >= min_length and not w.isdigit()]
            all_words.extend(words)

        # Contar frecuencia
        word_counts = Counter(all_words)

        # Retornar las más comunes
        return [word for word, _ in word_counts.most_common(3)]

    def _extract_keywords_from_filenames(
        self,
        doc_ids: List[str],
        labels: np.ndarray,
        cluster_id: int,
        metadata: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        Extrae keywords de nombres de archivos del cluster.

        Args:
            doc_ids: Lista de IDs de documentos
            labels: Array de etiquetas
            cluster_id: ID del cluster
            metadata: Metadata de documentos

        Returns:
            Lista de keywords
        """
        cluster_doc_ids = [doc_ids[i] for i, label in enumerate(labels) if label == cluster_id]
        filenames = [
            metadata.get(doc_id, {}).get("filename", "")
            for doc_id in cluster_doc_ids
            if doc_id in metadata
        ]

        keywords = self._find_common_words(filenames, min_length=3)
        return keywords[:10]  # Máximo 10 keywords
