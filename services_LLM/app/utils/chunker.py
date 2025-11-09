"""
Text chunking module.
Divide texto en chunks usando RecursiveCharacterTextSplitter de LangChain.
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

# Cargar variables de entorno
load_dotenv()

# Configuración por defecto
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: List[str] = None
) -> List[str]:
    """
    Divide texto en chunks usando RecursiveCharacterTextSplitter.

    Args:
        text: Texto a dividir
        chunk_size: Tamaño máximo de cada chunk en caracteres
        chunk_overlap: Número de caracteres de superposición entre chunks
        separators: Lista de separadores personalizados (None usa los por defecto)

    Returns:
        List[str]: Lista de chunks de texto

    Example:
        ```python
        text = "Este es un texto largo..."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        print(f"Se crearon {len(chunks)} chunks")
        ```
    """
    if not text or not text.strip():
        logger.warning("Texto vacío proporcionado para chunking")
        return []

    # Separadores por defecto optimizados para texto general
    if separators is None:
        separators = [
            "\n\n",  # Párrafos
            "\n",    # Líneas
            ". ",    # Oraciones
            ", ",    # Cláusulas
            " ",     # Palabras
            ""       # Caracteres
        ]

    try:
        # Crear el splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )

        # Dividir el texto
        chunks = text_splitter.split_text(text)

        logger.info(
            f"Texto dividido en {len(chunks)} chunks "
            f"(size={chunk_size}, overlap={chunk_overlap})"
        )

        # Log de estadísticas
        if chunks:
            avg_length = sum(len(chunk) for chunk in chunks) / len(chunks)
            min_length = min(len(chunk) for chunk in chunks)
            max_length = max(len(chunk) for chunk in chunks)

            logger.debug(
                f"Estadísticas de chunks: "
                f"promedio={avg_length:.0f}, min={min_length}, max={max_length}"
            )

        return chunks

    except Exception as e:
        logger.error(f"Error al dividir texto en chunks: {e}")
        raise


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: dict = None
) -> List[dict]:
    """
    Divide texto en chunks y retorna con metadata adjunta.

    Args:
        text: Texto a dividir
        chunk_size: Tamaño máximo de cada chunk
        chunk_overlap: Superposición entre chunks
        metadata: Metadata base a adjuntar a cada chunk

    Returns:
        List[dict]: Lista de diccionarios con 'text' y 'metadata'

    Example:
        ```python
        chunks = chunk_text_with_metadata(
            text="Texto largo...",
            metadata={"document_id": "123", "filename": "doc.pdf"}
        )
        for chunk in chunks:
            print(chunk['text'])
            print(chunk['metadata'])
        ```
    """
    # Obtener chunks básicos
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # Metadata base
    base_metadata = metadata or {}

    # Crear chunks con metadata
    chunks_with_metadata = []

    for idx, chunk_content in enumerate(chunks):
        chunk_metadata = {
            **base_metadata,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "chunk_size": len(chunk_content)
        }

        chunks_with_metadata.append({
            "text": chunk_content,
            "metadata": chunk_metadata
        })

    return chunks_with_metadata


def chunk_documents(
    documents: List[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[str]:
    """
    Divide múltiples documentos en chunks.

    Args:
        documents: Lista de textos/documentos
        chunk_size: Tamaño máximo de cada chunk
        chunk_overlap: Superposición entre chunks

    Returns:
        List[str]: Lista de todos los chunks de todos los documentos
    """
    all_chunks = []

    for idx, doc in enumerate(documents):
        logger.debug(f"Procesando documento {idx + 1}/{len(documents)}")
        doc_chunks = chunk_text(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)

    logger.info(
        f"Total: {len(all_chunks)} chunks de {len(documents)} documentos"
    )

    return all_chunks


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estima el número de tokens en un texto.
    Nota: Esto es una aproximación. Para conteo exacto usa tiktoken.

    Args:
        text: Texto a estimar
        chars_per_token: Promedio de caracteres por token (4 es una buena aproximación)

    Returns:
        int: Número estimado de tokens
    """
    return int(len(text) / chars_per_token)


def get_optimal_chunk_size(
    total_text_length: int,
    target_chunks: int = 10,
    min_size: int = 500,
    max_size: int = 2000
) -> int:
    """
    Calcula un tamaño de chunk óptimo basado en la longitud del texto.

    Args:
        total_text_length: Longitud total del texto
        target_chunks: Número aproximado de chunks deseados
        min_size: Tamaño mínimo de chunk
        max_size: Tamaño máximo de chunk

    Returns:
        int: Tamaño de chunk sugerido
    """
    calculated_size = total_text_length // target_chunks

    # Ajustar a límites
    optimal_size = max(min_size, min(calculated_size, max_size))

    logger.debug(
        f"Tamaño óptimo calculado: {optimal_size} "
        f"(texto: {total_text_length} chars, target: {target_chunks} chunks)"
    )

    return optimal_size


if __name__ == "__main__":
    # Test del módulo
    sample_text = """
    Este es un texto de prueba para el sistema de chunking.

    El sistema debe ser capaz de dividir textos largos en chunks más pequeños y manejables.
    Esto es importante para el procesamiento de embeddings y para evitar límites de tokens.

    Los chunks deben tener superposición para mantener el contexto entre fragmentos.
    Esto ayuda a que el sistema de recuperación funcione mejor.

    Cada chunk debe ser lo suficientemente grande para tener significado,
    pero lo suficientemente pequeño para ser procesado eficientemente.
    """ * 10  # Repetir para tener texto más largo

    print(f"📝 Texto original: {len(sample_text)} caracteres")
    print(f"🔢 Tokens estimados: {estimate_tokens(sample_text)}")

    # Test chunking básico
    chunks = chunk_text(sample_text, chunk_size=300, chunk_overlap=50)
    print(f"✂️ Chunks creados: {len(chunks)}")

    # Mostrar primeros chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ({len(chunk)} chars) ---")
        print(chunk[:100] + "...")

    # Test con metadata
    chunks_meta = chunk_text_with_metadata(
        sample_text,
        chunk_size=300,
        metadata={"test": True, "source": "demo"}
    )
    print(f"\n📦 Chunks con metadata: {len(chunks_meta)}")
    print(f"Ejemplo metadata: {chunks_meta[0]['metadata']}")
