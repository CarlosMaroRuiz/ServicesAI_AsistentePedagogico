"""
Adaptador para UMAP (Uniform Manifold Approximation and Projection).

Reduce dimensionalidad de embeddings manteniendo estructura semántica.
"""
import numpy as np
from typing import Optional
import umap
from loguru import logger

from app.core.config import settings


class UMAPAdapter:
    """Adaptador para reducción dimensional con UMAP."""

    def __init__(
        self,
        n_components: int | None = None,
        n_neighbors: int | None = None,
        metric: str | None = None,
        min_dist: float | None = None,
        random_state: int = 42,
    ):
        """
        Inicializa el adaptador UMAP.

        Args:
            n_components: Dimensiones finales (5 para clustering, 2 para viz)
            n_neighbors: Tamaño de vecindad local
            metric: Métrica de distancia ('cosine' para embeddings)
            min_dist: Distancia mínima entre puntos en espacio reducido
            random_state: Semilla para reproducibilidad
        """
        self.n_components = n_components or settings.umap_n_components_cluster
        self.n_neighbors = n_neighbors or settings.umap_n_neighbors
        self.metric = metric or settings.umap_metric
        self.min_dist = min_dist or settings.umap_min_dist
        self.random_state = random_state
        self.reducer = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Reduce dimensionalidad de embeddings.

        Args:
            embeddings: Array de embeddings (N x D), típicamente 384D

        Returns:
            Array de embeddings reducidos (N x n_components)
        """
        logger.info(
            f"Reduciendo dimensionalidad con UMAP: {embeddings.shape[1]}D → {self.n_components}D"
        )
        logger.debug(
            f"Parámetros: n_neighbors={self.n_neighbors}, metric={self.metric}, min_dist={self.min_dist}"
        )

        try:
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                min_dist=self.min_dist,
                random_state=self.random_state,
                verbose=False,
            )

            reduced_embeddings = self.reducer.fit_transform(embeddings)

            logger.info(
                f"✅ Reducción completada: {embeddings.shape} → {reduced_embeddings.shape}"
            )

            return reduced_embeddings

        except Exception as e:
            logger.error(f"Error en UMAP: {str(e)}", exc_info=True)
            raise

    def transform(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Transforma nuevos embeddings usando el modelo ajustado.

        Args:
            new_embeddings: Nuevos embeddings a reducir

        Returns:
            Embeddings reducidos
        """
        if self.reducer is None:
            raise ValueError("Debe ejecutar fit_transform() primero")

        try:
            return self.reducer.transform(new_embeddings)
        except Exception as e:
            logger.error(f"Error transformando nuevos embeddings: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def create_for_clustering() -> "UMAPAdapter":
        """
        Crea un adaptador UMAP configurado para clustering.

        Returns:
            UMAPAdapter con 5 componentes
        """
        return UMAPAdapter(n_components=settings.umap_n_components_cluster)

    @staticmethod
    def create_for_visualization() -> "UMAPAdapter":
        """
        Crea un adaptador UMAP configurado para visualización 2D.

        Returns:
            UMAPAdapter con 2 componentes
        """
        return UMAPAdapter(n_components=settings.umap_n_components_viz)

    def get_embedding_info(self, reduced_embeddings: np.ndarray) -> dict:
        """
        Obtiene información sobre los embeddings reducidos.

        Args:
            reduced_embeddings: Embeddings después de reducción

        Returns:
            Diccionario con estadísticas
        """
        return {
            "shape": reduced_embeddings.shape,
            "n_samples": reduced_embeddings.shape[0],
            "n_dimensions": reduced_embeddings.shape[1],
            "mean": float(np.mean(reduced_embeddings)),
            "std": float(np.std(reduced_embeddings)),
            "min": float(np.min(reduced_embeddings)),
            "max": float(np.max(reduced_embeddings)),
        }
