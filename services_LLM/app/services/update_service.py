"""
Servicio de actualización de documentos.
Actualiza documentos existentes y sus embeddings.
"""

import os
from typing import Optional
from loguru import logger
from psycopg.types.json import Json

from app.services.delete_service import delete_document_chunks
from app.services.ingest_service import process_and_store_pdf
from app.db.connection import get_db_connection


def update_document(
    document_id: str,
    new_file_path: str,
    new_filename: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> dict:
    """
    Actualiza un documento existente.

    Proceso:
    1. Verifica que el documento existe
    2. Elimina los chunks antiguos del vector store
    3. Procesa el nuevo PDF
    4. Actualiza la metadata del documento

    Args:
        document_id: UUID del documento a actualizar
        new_file_path: Ruta al nuevo archivo PDF
        new_filename: Nombre del nuevo archivo (opcional)
        chunk_size: Tamaño de chunks
        chunk_overlap: Superposición de chunks

    Returns:
        dict: Información de la actualización

    Raises:
        ValueError: Si el documento no existe o hay errores de validación
        Exception: Otros errores de procesamiento
    """
    try:
        # 1. Verificar que el documento existe
        logger.info(f"Actualizando documento: {document_id}")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, user_id, filename, total_chunks
                    FROM documents
                    WHERE id = %s
                """, (document_id,))

                doc = cur.fetchone()

                if not doc:
                    raise ValueError(f"Documento no encontrado: {document_id}")

        user_id = doc['user_id']
        old_filename = doc['filename']
        old_chunks = doc['total_chunks']

        logger.info(f"Documento encontrado: {old_filename} (user: {user_id}, chunks: {old_chunks})")

        # 2. Eliminar chunks antiguos del vector store
        logger.info("Eliminando chunks antiguos del vector store")
        delete_document_chunks(document_id, user_id)

        # 3. Marcar documento como 'updating'
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents
                    SET status = 'updating', last_update = NOW()
                    WHERE id = %s
                """, (document_id,))
                conn.commit()

        # 4. Procesar nuevo PDF (esto creará un nuevo documento)
        filename = new_filename or os.path.basename(new_file_path)

        new_doc_result = process_and_store_pdf(
            user_id=user_id,
            file_path=new_file_path,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        new_document_id = new_doc_result['document_id']

        # 5. Eliminar el documento antiguo y transferir metadata
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Obtener metadata del documento nuevo
                cur.execute("""
                    SELECT metadata FROM documents WHERE id = %s
                """, (new_document_id,))
                new_metadata = cur.fetchone()['metadata']

                # Actualizar el documento antiguo con nueva info
                cur.execute("""
                    UPDATE documents
                    SET
                        filename = %s,
                        file_path = %s,
                        file_size_bytes = %s,
                        total_chunks = %s,
                        status = 'active',
                        last_update = NOW(),
                        metadata = %s
                    WHERE id = %s
                """, (
                    filename,
                    new_file_path,
                    new_doc_result.get('file_size_mb', 0) * 1024 * 1024,
                    new_doc_result['chunks_created'],
                    Json(new_metadata),
                    document_id
                ))

                # Eliminar el documento temporal nuevo
                cur.execute("""
                    DELETE FROM documents WHERE id = %s
                """, (new_document_id,))

                conn.commit()

        logger.info(f"Documento actualizado exitosamente: {document_id}")

        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "old_chunks": old_chunks,
            "new_chunks": new_doc_result['chunks_created'],
            "file_size_mb": new_doc_result['file_size_mb'],
            "message": f"Documento actualizado: {old_chunks} -> {new_doc_result['chunks_created']} chunks"
        }

    except Exception as e:
        logger.error(f"Error al actualizar documento {document_id}: {e}")

        # Intentar revertir el estado
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE documents
                        SET status = 'error', metadata = metadata || %s::jsonb
                        WHERE id = %s
                    """, (Json({"update_error": str(e)}), document_id))
                    conn.commit()
        except:
            pass

        raise


def update_document_metadata(
    document_id: str,
    metadata_updates: dict
) -> dict:
    """
    Actualiza solo la metadata de un documento sin reprocesarlo.

    Args:
        document_id: UUID del documento
        metadata_updates: Diccionario con campos a actualizar

    Returns:
        dict: Resultado de la actualización
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Verificar que existe
                cur.execute("SELECT id FROM documents WHERE id = %s", (document_id,))
                if not cur.fetchone():
                    raise ValueError(f"Documento no encontrado: {document_id}")

                # Actualizar metadata
                cur.execute("""
                    UPDATE documents
                    SET metadata = metadata || %s::jsonb, last_update = NOW()
                    WHERE id = %s
                    RETURNING metadata
                """, (Json(metadata_updates), document_id))

                updated_metadata = cur.fetchone()['metadata']
                conn.commit()

        logger.info(f"Metadata actualizada para documento {document_id}")

        return {
            "status": "success",
            "document_id": document_id,
            "updated_metadata": updated_metadata
        }

    except Exception as e:
        logger.error(f"Error al actualizar metadata: {e}")
        raise


def rename_document(document_id: str, new_filename: str) -> dict:
    """
    Renombra un documento.

    Args:
        document_id: UUID del documento
        new_filename: Nuevo nombre

    Returns:
        dict: Resultado
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents
                    SET filename = %s, last_update = NOW()
                    WHERE id = %s
                    RETURNING filename
                """, (new_filename, document_id))

                if cur.rowcount == 0:
                    raise ValueError(f"Documento no encontrado: {document_id}")

                conn.commit()

        logger.info(f"Documento renombrado: {document_id} -> {new_filename}")

        return {
            "status": "success",
            "document_id": document_id,
            "new_filename": new_filename
        }

    except Exception as e:
        logger.error(f"Error al renombrar documento: {e}")
        raise


if __name__ == "__main__":
    # Test del servicio
    import sys

    if len(sys.argv) > 2:
        test_doc_id = sys.argv[1]
        test_file_path = sys.argv[2]

        print(f"🔄 Actualizando documento")
        print(f"ID: {test_doc_id}")
        print(f"Nuevo archivo: {test_file_path}")

        result = update_document(
            document_id=test_doc_id,
            new_file_path=test_file_path
        )

        print(f"Resultado: {result}")
    else:
        print("Uso: python update_service.py <document_id> <new_pdf_path>")
