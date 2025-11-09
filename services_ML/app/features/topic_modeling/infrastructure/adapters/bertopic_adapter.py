"""
Adaptador para BERTopic (Topic Modeling basado en BERT).

Extrae temas principales de documentos usando embeddings pre-calculados.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from bertopic import BERTopic
from loguru import logger

from app.core.config import settings


class BERTopicAdapter:
    """Adaptador para extracción de temas con BERTopic."""

    def __init__(
        self,
        language: str | None = None,
        top_n_words: int | None = None,
        nr_topics: int | str | None = "auto",
        calculate_probabilities: bool | None = None,
    ):
        """
        Inicializa el adaptador BERTopic.

        Args:
            language: Idioma para stopwords ('spanish')
            top_n_words: Número de palabras clave por tema
            nr_topics: Número de temas ('auto' o entero específico)
            calculate_probabilities: Si calcular probabilidades (más lento)
        """
        self.language = language or settings.topic_language
        self.top_n_words = top_n_words or settings.top_n_words
        self.nr_topics = nr_topics
        self.calculate_probabilities = (
            calculate_probabilities
            if calculate_probabilities is not None
            else settings.calculate_probabilities
        )
        self.model = None
        self.topics = None
        self.probabilities = None

    def fit(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        umap_reducer=None,
        hdbscan_clusterer=None,
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        Ajusta el modelo de temas a los documentos.

        Args:
            documents: Lista de textos de documentos
            embeddings: Embeddings pre-calculados (sentence-transformers)
            umap_reducer: Reductor UMAP ya ajustado (opcional)
            hdbscan_clusterer: Clusterer HDBSCAN ya ajustado (opcional)

        Returns:
            Tuple de (topics, probabilities)
        """
        logger.info(f"Extrayendo temas de {len(documents)} documentos con BERTopic")
        logger.debug(
            f"Parámetros: language={self.language}, top_n_words={self.top_n_words}, nr_topics={self.nr_topics}"
        )

        try:
            # Crear modelo BERTopic
            self.model = BERTopic(
                language=self.language,
                top_n_words=self.top_n_words,
                nr_topics=self.nr_topics,
                calculate_probabilities=self.calculate_probabilities,
                verbose=False,
                # Usar UMAP y HDBSCAN externos si se proporcionan
                umap_model=umap_reducer,
                hdbscan_model=hdbscan_clusterer,
            )

            # Ajustar modelo
            if self.calculate_probabilities:
                self.topics, self.probabilities = self.model.fit_transform(
                    documents, embeddings=embeddings
                )
            else:
                self.topics = self.model.fit_transform(documents, embeddings=embeddings)[0]
                self.probabilities = None

            # Estadísticas
            unique_topics = set(self.topics)
            num_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
            num_outliers = self.topics.count(-1)

            logger.info(f"✅ Extracción de temas completada:")
            logger.info(f"   - Temas encontrados: {num_topics}")
            logger.info(f"   - Documentos outliers: {num_outliers}")

            return self.topics, self.probabilities

        except Exception as e:
            logger.error(f"Error en BERTopic: {str(e)}", exc_info=True)
            raise

    def get_topic_info(self) -> List[Dict[str, Any]]:
        """
        Obtiene información de todos los temas.

        Returns:
            Lista de diccionarios con info de cada tema
        """
        if self.model is None:
            raise ValueError("Debe ejecutar fit() primero")

        topic_info = []
        topic_ids = sorted(set(self.topics))

        for topic_id in topic_ids:
            if topic_id == -1:
                continue  # Saltar outliers

            # Obtener palabras clave del tema
            topic_words = self.model.get_topic(topic_id)
            keywords = [word for word, _ in topic_words[:self.top_n_words]]

            # Generar etiqueta descriptiva
            label = self._generate_topic_label(topic_id, keywords)

            # Contar documentos en este tema
            document_count = self.topics.count(topic_id)

            topic_info.append(
                {
                    "topic_id": topic_id,
                    "label": label,
                    "keywords": keywords,
                    "document_count": document_count,
                    "word_scores": topic_words[:self.top_n_words],
                }
            )

        return topic_info

    def _generate_topic_label(self, topic_id: int, keywords: List[str]) -> str:
        """
        Genera una etiqueta descriptiva para el tema.

        Args:
            topic_id: ID del tema
            keywords: Palabras clave del tema

        Returns:
            Etiqueta descriptiva
        """
        if len(keywords) == 0:
            return f"Tema {topic_id}"

        # Tomar las 3 palabras más importantes
        top_words = keywords[:3]
        label = " - ".join(top_words).title()

        return f"{label}"

    def get_representative_docs(
        self, documents: List[str], topic_id: int, n_docs: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Obtiene documentos más representativos de un tema.

        Args:
            documents: Lista de todos los documentos
            topic_id: ID del tema
            n_docs: Número de documentos a retornar

        Returns:
            Lista de documentos representativos
        """
        if self.model is None:
            raise ValueError("Debe ejecutar fit() primero")

        # Encontrar índices de documentos de este tema
        topic_indices = [i for i, t in enumerate(self.topics) if t == topic_id]

        if len(topic_indices) == 0:
            return []

        # Si hay probabilidades, ordenar por probabilidad
        if self.probabilities is not None:
            probs = [self.probabilities[i][topic_id] for i in topic_indices]
            sorted_indices = sorted(
                zip(topic_indices, probs), key=lambda x: x[1], reverse=True
            )
            top_indices = [idx for idx, _ in sorted_indices[:n_docs]]
        else:
            # Si no hay probabilidades, tomar los primeros N
            top_indices = topic_indices[:n_docs]

        # Construir resultado
        representative_docs = []
        for idx in top_indices:
            doc_info = {
                "index": idx,
                "text": documents[idx][:500],  # Primeros 500 caracteres
                "topic_id": topic_id,
            }

            if self.probabilities is not None:
                doc_info["probability"] = float(self.probabilities[idx][topic_id])

            representative_docs.append(doc_info)

        return representative_docs

    def reduce_topics(self, nr_topics: int) -> None:
        """
        Reduce el número de temas combinando similares.

        Args:
            nr_topics: Número de temas deseado
        """
        if self.model is None:
            raise ValueError("Debe ejecutar fit() primero")

        logger.info(f"Reduciendo temas a {nr_topics}")
        self.model.reduce_topics(nr_topics=nr_topics)
        self.topics = self.model.topics_

    def get_topic_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas generales de los temas.

        Returns:
            Diccionario con estadísticas
        """
        if self.topics is None:
            raise ValueError("Debe ejecutar fit() primero")

        unique_topics = set(self.topics)
        num_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
        num_outliers = self.topics.count(-1)

        # Distribución de documentos por tema
        topic_distribution = {}
        for topic_id in unique_topics:
            if topic_id != -1:
                topic_distribution[topic_id] = self.topics.count(topic_id)

        return {
            "num_topics": num_topics,
            "num_outliers": num_outliers,
            "total_documents": len(self.topics),
            "outlier_percentage": (num_outliers / len(self.topics) * 100),
            "topic_distribution": topic_distribution,
            "avg_docs_per_topic": (
                sum(topic_distribution.values()) / len(topic_distribution)
                if topic_distribution
                else 0
            ),
        }

    def update_topics(
        self, new_documents: List[str], new_embeddings: np.ndarray
    ) -> List[int]:
        """
        Actualiza el modelo con nuevos documentos.

        Args:
            new_documents: Nuevos documentos
            new_embeddings: Embeddings de nuevos documentos

        Returns:
            Temas asignados a nuevos documentos
        """
        if self.model is None:
            raise ValueError("Debe ejecutar fit() primero")

        try:
            new_topics = self.model.transform(new_documents, embeddings=new_embeddings)[0]
            return new_topics
        except Exception as e:
            logger.error(f"Error actualizando temas: {str(e)}", exc_info=True)
            raise
