"""
Adaptador para HDBSCAN (Hierarchical Density-Based Spatial Clustering).

Proporciona clustering automático sin necesidad de especificar número de clusters.
"""
import numpy as np
from typing import Dict, Any, Tuple, List
import hdbscan
from loguru import logger

from app.core.config import settings


class HDBSCANAdapter:
    """Adaptador para algoritmo de clustering HDBSCAN."""

    def __init__(
        self,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        cluster_selection_method: str | None = None,
    ):
        """
        Inicializa el adaptador HDBSCAN.

        Args:
            min_cluster_size: Tamaño mínimo del cluster (default: settings.min_cluster_size)
            min_samples: Muestras mínimas para densidad (default: settings.min_samples)
            cluster_selection_method: Método de selección (default: settings.cluster_selection_method)
        """
        self.min_cluster_size = min_cluster_size or settings.min_cluster_size
        self.min_samples = min_samples or settings.min_samples
        self.cluster_selection_method = cluster_selection_method or settings.cluster_selection_method
        self.clusterer = None
        self.labels = None
        self.probabilities = None

    def fit(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta el modelo de clustering a los embeddings.

        Args:
            embeddings: Array de embeddings (N x D)

        Returns:
            Tuple de (labels, probabilities)
            - labels: Array de etiquetas de cluster (-1 para outliers)
            - probabilities: Array de probabilidades de pertenencia
        """
        logger.info(f"Ejecutando HDBSCAN en {len(embeddings)} documentos")
        logger.debug(
            f"Parámetros: min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}"
        )

        try:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_method=self.cluster_selection_method,
                metric="euclidean",  # Usar euclidean después de UMAP
                prediction_data=True,  # Habilitar predicciones
            )

            self.labels = self.clusterer.fit_predict(embeddings)
            self.probabilities = self.clusterer.probabilities_

            # Estadísticas
            unique_labels = np.unique(self.labels)
            num_clusters = len(unique_labels[unique_labels >= 0])
            num_outliers = np.sum(self.labels == -1)

            logger.info(f"✅ Clustering completado:")
            logger.info(f"   - Clusters encontrados: {num_clusters}")
            logger.info(f"   - Outliers: {num_outliers} ({num_outliers/len(self.labels)*100:.1f}%)")

            return self.labels, self.probabilities

        except Exception as e:
            logger.error(f"Error en HDBSCAN: {str(e)}", exc_info=True)
            raise

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del clustering.

        Returns:
            Diccionario con estadísticas
        """
        if self.labels is None:
            raise ValueError("Debe ejecutar fit() primero")

        unique_labels = np.unique(self.labels)
        clusters = unique_labels[unique_labels >= 0]

        stats = {
            "num_clusters": len(clusters),
            "num_outliers": int(np.sum(self.labels == -1)),
            "total_documents": len(self.labels),
            "outlier_percentage": float(np.sum(self.labels == -1) / len(self.labels) * 100),
            "cluster_sizes": {},
            "cluster_ids": clusters.tolist(),
        }

        # Tamaño de cada cluster
        for cluster_id in clusters:
            size = int(np.sum(self.labels == cluster_id))
            stats["cluster_sizes"][int(cluster_id)] = size

        return stats

    def get_cluster_centroids(self, embeddings: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Calcula centroides de cada cluster.

        Args:
            embeddings: Embeddings originales reducidos

        Returns:
            Diccionario {cluster_id: centroid_vector}
        """
        if self.labels is None:
            raise ValueError("Debe ejecutar fit() primero")

        centroids = {}
        unique_labels = np.unique(self.labels)
        clusters = unique_labels[unique_labels >= 0]

        for cluster_id in clusters:
            mask = self.labels == cluster_id
            cluster_embeddings = embeddings[mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids[int(cluster_id)] = centroid

        return centroids

    def get_representative_points(
        self, embeddings: np.ndarray, n_points: int = 3
    ) -> Dict[int, List[int]]:
        """
        Obtiene puntos más representativos de cada cluster.

        Args:
            embeddings: Embeddings reducidos
            n_points: Número de puntos representativos por cluster

        Returns:
            Diccionario {cluster_id: [indices de puntos representativos]}
        """
        if self.labels is None:
            raise ValueError("Debe ejecutar fit() primero")

        representative_points = {}
        unique_labels = np.unique(self.labels)
        clusters = unique_labels[unique_labels >= 0]

        for cluster_id in clusters:
            mask = self.labels == cluster_id
            cluster_indices = np.where(mask)[0]
            cluster_embeddings = embeddings[mask]

            # Calcular centroide
            centroid = np.mean(cluster_embeddings, axis=0)

            # Encontrar puntos más cercanos al centroide
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_indices = np.argsort(distances)[:n_points]

            # Mapear a índices originales
            original_indices = cluster_indices[closest_indices]
            representative_points[int(cluster_id)] = original_indices.tolist()

        return representative_points

    def predict(self, new_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice cluster para nuevos embeddings.

        Args:
            new_embeddings: Nuevos embeddings a clasificar

        Returns:
            Tuple de (labels, probabilities)
        """
        if self.clusterer is None:
            raise ValueError("Debe ejecutar fit() primero")

        try:
            labels, probabilities = hdbscan.approximate_predict(self.clusterer, new_embeddings)
            return labels, probabilities
        except Exception as e:
            logger.error(f"Error prediciendo clusters: {str(e)}", exc_info=True)
            raise
