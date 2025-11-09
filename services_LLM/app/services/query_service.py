"""
Servicio de consulta y recuperación de contexto.
Busca chunks relevantes usando similarity search en pgvector.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from loguru import logger

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# Cargar variables de entorno
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings_model():
    """
    Obtiene el modelo de embeddings (mismo que en ingest_service).

    Returns:
        HuggingFaceEmbeddings: Modelo de embeddings configurado
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def get_vectorstore(collection_name: str = "documents") -> PGVector:
    """
    Obtiene una instancia del vectorstore.

    Args:
        collection_name: Nombre de la colección en pgvector

    Returns:
        PGVector: Instancia del vectorstore
    """
    embeddings = get_embeddings_model()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    return vectorstore


def get_relevant_chunks(
    query: str,
    user_id: Optional[str] = None,
    k: int = 3,
    score_threshold: Optional[float] = None,
    filter_metadata: Optional[dict] = None
) -> List[Document]:
    """
    Busca chunks relevantes para una consulta.

    Args:
        query: Pregunta o consulta del usuario
        user_id: ID del usuario (para filtrar documentos)
        k: Número de chunks a retornar
        score_threshold: Umbral mínimo de similitud (0-1)
        filter_metadata: Filtros adicionales de metadata

    Returns:
        List[Document]: Lista de documentos relevantes con metadata

    Example:
        ```python
        chunks = get_relevant_chunks(
            query="¿Qué dice el contrato sobre pagos?",
            user_id="user123",
            k=5
        )
        for chunk in chunks:
            print(chunk.page_content)
            print(chunk.metadata)
        ```
    """
    try:
        # Construir filtros
        filters = {}
        if user_id:
            filters["user_id"] = {"$eq": user_id}

        if filter_metadata:
            filters.update(filter_metadata)

        logger.info(f"Buscando chunks relevantes para: '{query[:50]}...'")
        logger.debug(f"Filtros: {filters}, k={k}")

        # Obtener vectorstore
        collection_name = f"documents_{user_id}" if user_id else "documents"
        vectorstore = get_vectorstore(collection_name)

        # Búsqueda por similitud
        if score_threshold is not None:
            # Búsqueda con threshold
            chunks = vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filters if filters else None
            )
            # Filtrar por threshold
            chunks = [(doc, score) for doc, score in chunks if score >= score_threshold]
            documents = [doc for doc, score in chunks]
        else:
            # Búsqueda simple
            documents = vectorstore.similarity_search(
                query,
                k=k,
                filter=filters if filters else None
            )

        logger.info(f"Encontrados {len(documents)} chunks relevantes")

        return documents

    except Exception as e:
        logger.error(f"Error en búsqueda de chunks: {e}")
        return []


def get_relevant_chunks_with_scores(
    query: str,
    user_id: Optional[str] = None,
    k: int = 3
) -> List[tuple[Document, float]]:
    """
    Busca chunks relevantes y retorna con sus scores de similitud.

    Args:
        query: Pregunta o consulta del usuario
        user_id: ID del usuario
        k: Número de chunks a retornar

    Returns:
        List[tuple[Document, float]]: Lista de (documento, score)
    """
    try:
        filters = {}
        if user_id:
            filters["user_id"] = {"$eq": user_id}

        collection_name = f"documents_{user_id}" if user_id else "documents"
        vectorstore = get_vectorstore(collection_name)

        results = vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filters if filters else None
        )

        logger.info(f"Encontrados {len(results)} chunks con scores")

        return results

    except Exception as e:
        logger.error(f"Error en búsqueda con scores: {e}")
        return []


def create_retriever(
    user_id: Optional[str] = None,
    k: int = 3,
    search_type: str = "similarity"
):
    """
    Crea un retriever de LangChain para usar en chains.

    Args:
        user_id: ID del usuario (para filtrar)
        k: Número de documentos a recuperar
        search_type: Tipo de búsqueda ("similarity", "mmr", "similarity_score_threshold")

    Returns:
        VectorStoreRetriever: Retriever configurado

    Example:
        ```python
        retriever = create_retriever(user_id="user123", k=5)
        docs = retriever.get_relevant_documents("¿Qué dice el contrato?")
        ```
    """
    collection_name = f"documents_{user_id}" if user_id else "documents"
    vectorstore = get_vectorstore(collection_name)

    search_kwargs = {"k": k}

    # Agregar filtro de user_id si se proporciona
    if user_id:
        search_kwargs["filter"] = {"user_id": {"$eq": user_id}}

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

    logger.info(f"Retriever creado: type={search_type}, k={k}, user={user_id}")

    return retriever


def format_chunks_for_context(chunks: List[Document]) -> str:
    """
    Formatea chunks para usarlos como contexto en el prompt.

    Args:
        chunks: Lista de documentos

    Returns:
        str: Contexto formateado
    """
    if not chunks:
        return "No se encontró información relevante."

    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        filename = chunk.metadata.get('filename', 'Desconocido')
        chunk_idx = chunk.metadata.get('chunk_index', '?')

        context_parts.append(
            f"[Fuente {i} - {filename}, chunk {chunk_idx}]\n"
            f"{chunk.page_content}\n"
        )

    return "\n---\n".join(context_parts)


def search_in_documents(
    query: str,
    user_id: str,
    document_ids: Optional[List[str]] = None,
    k: int = 3
) -> List[Document]:
    """
    Busca en documentos específicos.

    Args:
        query: Consulta de búsqueda
        user_id: ID del usuario
        document_ids: Lista de IDs de documentos específicos
        k: Número de resultados

    Returns:
        List[Document]: Chunks relevantes
    """
    filters = {"user_id": {"$eq": user_id}}

    if document_ids:
        # Filtrar por documentos específicos
        filters["document_id"] = {"$in": document_ids}

    collection_name = f"documents_{user_id}"
    vectorstore = get_vectorstore(collection_name)

    documents = vectorstore.similarity_search(
        query,
        k=k,
        filter=filters
    )

    return documents


if __name__ == "__main__":
    # Test del servicio
    import sys

    if len(sys.argv) > 2:
        test_user_id = sys.argv[1]
        test_query = " ".join(sys.argv[2:])

        print(f"🔍 Buscando: '{test_query}'")
        print(f"👤 Usuario: {test_user_id}")

        chunks = get_relevant_chunks(
            query=test_query,
            user_id=test_user_id,
            k=3
        )

        print(f"\n📄 Encontrados {len(chunks)} chunks:\n")

        for i, chunk in enumerate(chunks, 1):
            print(f"--- Chunk {i} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Contenido: {chunk.page_content[:200]}...")
            print()

        # Test con scores
        print("\n📊 Con scores de similitud:")
        chunks_scored = get_relevant_chunks_with_scores(
            query=test_query,
            user_id=test_user_id,
            k=3
        )

        for i, (chunk, score) in enumerate(chunks_scored, 1):
            print(f"{i}. Score: {score:.4f} | {chunk.metadata.get('filename', 'N/A')}")

    else:
        print("Uso: python query_service.py <user_id> <consulta>")
