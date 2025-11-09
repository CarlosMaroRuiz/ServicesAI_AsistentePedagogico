"""
Servicio de ingesta de documentos.
Procesa PDFs, extrae texto, crea chunks y almacena embeddings en pgvector.
"""

import os
import uuid
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from psycopg.types.json import Json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

from app.utils.text_extractor import extract_text_from_pdf, get_pdf_metadata, validate_pdf_file
from app.utils.chunker import chunk_text_with_metadata
from app.db.connection import get_db_connection

# Cargar variables de entorno
load_dotenv()

# Configuración
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))


def get_embeddings_model():
    """
    Obtiene el modelo de embeddings.

    Returns:
        HuggingFaceEmbeddings: Modelo de embeddings configurado
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def process_and_store_pdf(
    user_id: str,
    file_path: str,
    filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> dict:
    """
    Procesa un documento (PDF o TXT) y almacena sus chunks con embeddings en pgvector.

    Args:
        user_id: ID del usuario
        file_path: Ruta al archivo
        filename: Nombre del archivo
        chunk_size: Tamaño de cada chunk
        chunk_overlap: Superposición entre chunks

    Returns:
        dict: Información del documento procesado

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo es inválido
        Exception: Otros errores de procesamiento
    """
    try:
        is_txt = filename.endswith('.txt')

        if is_txt:
            # Procesar archivo TXT
            logger.info(f"Procesando archivo TXT: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Metadata simple para TXT
            import os
            file_size = os.path.getsize(file_path)
            pdf_metadata = {
                "filename": filename,
                "file_size_bytes": file_size,
                "pages": 1,
                "file_type": "text/plain"
            }
        else:
            # 1. Validar el PDF
            logger.info(f"Validando PDF: {filename}")
            is_valid, validation_message = validate_pdf_file(file_path, MAX_FILE_SIZE_MB)

            if not is_valid:
                raise ValueError(f"PDF inválido: {validation_message}")

            # 2. Obtener metadata del PDF
            pdf_metadata = get_pdf_metadata(file_path)
            logger.info(f"PDF metadata: {pdf_metadata}")

            # 3. Extraer texto
            logger.info(f"Extrayendo texto de {filename}")
            text = extract_text_from_pdf(file_path)

        if not text or len(text.strip()) < 10:
            raise ValueError("El PDF no contiene texto extraíble suficiente")

        # 4. Crear documento en la base de datos
        logger.info(f"Creando registro de documento en DB")
        document_id = None

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (user_id, filename, file_path, file_size_bytes, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    user_id,
                    filename,
                    file_path,
                    pdf_metadata.get('file_size_bytes', 0),
                    Json(pdf_metadata)
                ))
                result = cur.fetchone()
                document_id = result['id']
                conn.commit()

        logger.info(f"Documento creado con ID: {document_id}")

        # 5. Dividir en chunks
        logger.info(f"Dividiendo texto en chunks (size={chunk_size}, overlap={chunk_overlap})")
        chunks_data = chunk_text_with_metadata(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={
                "document_id": str(document_id),
                "user_id": user_id,
                "filename": filename,
                "source": "pdf"
            }
        )

        logger.info(f"Creados {len(chunks_data)} chunks")

        # 6. Preparar documentos para LangChain
        documents = []
        for chunk_data in chunks_data:
            doc = Document(
                page_content=chunk_data['text'],
                metadata=chunk_data['metadata']
            )
            documents.append(doc)

        # 7. Crear embeddings y almacenar en pgvector
        logger.info(f"Creando embeddings y almacenando en pgvector")
        embeddings = get_embeddings_model()

        # Nombre de colección único por usuario (opcional)
        collection_name = f"documents_{user_id}" if user_id else "documents"

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=DATABASE_URL,
            use_jsonb=True,
        )

        # Agregar documentos al vector store
        ids = vectorstore.add_documents(documents)
        logger.info(f"Almacenados {len(ids)} chunks en pgvector")

        # 8. Actualizar el contador de chunks en la tabla documents
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents
                    SET total_chunks = %s, last_update = NOW()
                    WHERE id = %s
                """, (len(chunks_data), document_id))
                conn.commit()

        # 9. Retornar resultado
        return {
            "status": "success",
            "document_id": str(document_id),
            "filename": filename,
            "chunks_created": len(chunks_data),
            "file_size_mb": round(pdf_metadata.get('file_size_bytes', 0) / (1024 * 1024), 2),
            "total_characters": len(text),
            "metadata": pdf_metadata
        }

    except Exception as e:
        logger.error(f"Error al procesar PDF {filename}: {str(e)}")
        # Si se creó el documento, marcarlo como error
        if document_id:
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE documents
                            SET status = 'error', metadata = metadata || %s::jsonb
                            WHERE id = %s
                        """, (Json({"error": str(e)}), document_id))
                        conn.commit()
            except:
                pass
        raise


def get_document_by_id(document_id: str) -> Optional[dict]:
    """
    Obtiene información de un documento por su ID.

    Args:
        document_id: UUID del documento

    Returns:
        dict o None: Información del documento
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM documents WHERE id = %s
                """, (document_id,))
                return cur.fetchone()
    except Exception as e:
        logger.error(f"Error al obtener documento {document_id}: {e}")
        return None


def list_user_documents(user_id: str, status: str = "active") -> list:
    """
    Lista todos los documentos de un usuario.

    Args:
        user_id: ID del usuario
        status: Filtro por estado (active, inactive, deleted)

    Returns:
        list: Lista de documentos
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM documents
                    WHERE user_id = %s AND status = %s
                    ORDER BY upload_date DESC
                """, (user_id, status))
                return cur.fetchall()
    except Exception as e:
        logger.error(f"Error al listar documentos del usuario {user_id}: {e}")
        return []


if __name__ == "__main__":
    # Test del servicio
    import sys

    if len(sys.argv) > 2:
        test_user_id = sys.argv[1]
        test_pdf_path = sys.argv[2]

        print(f"🧪 Testeando ingesta de PDF")
        print(f"Usuario: {test_user_id}")
        print(f"Archivo: {test_pdf_path}")

        result = process_and_store_pdf(
            user_id=test_user_id,
            file_path=test_pdf_path,
            filename=Path(test_pdf_path).name
        )

        print(f"Resultado: {result}")
    else:
        print("Uso: python ingest_service.py <user_id> <pdf_path>")
