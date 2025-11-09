

import os
from typing import Optional
from loguru import logger
from dotenv import load_dotenv

from langchain_postgres import PGVector
from app.services.ingest_service import get_embeddings_model
from app.db.connection import get_db_connection

# Cargar variables de entorno
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def delete_document_chunks(document_id: str, user_id: str) -> int:
    """
    Elimina los chunks de un documento del vector store.

    Args:
        document_id: UUID del documento
        user_id: ID del usuario (para determinar la colección)

    Returns:
        int: Número de chunks eliminados
    """
    try:
        logger.info(f"Eliminando chunks del documento {document_id} del vector store")

        embeddings = get_embeddings_model()
        collection_name = f"documents_{user_id}"

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=DATABASE_URL,
            use_jsonb=True,
        )

        # Eliminar documentos por metadata
        # Nota: PGVector delete funciona con IDs, necesitamos buscar primero
        docs = vectorstore.similarity_search(
            "",
            k=1000,  # Número grande para obtener todos
            filter={"document_id": {"$eq": document_id}}
        )

        if not docs:
            logger.warning(f"No se encontraron chunks para documento {document_id}")
            return 0

        # Extraer IDs si están disponibles
        # Como workaround, eliminaremos usando SQL directo
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Determinar el nombre de tabla de la colección
                # langchain_pg_embedding es el nombre por defecto
                cur.execute("""
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'document_id' = %s
                """, (document_id,))

                deleted_count = cur.rowcount
                conn.commit()

        logger.info(f"Eliminados {deleted_count} chunks del vector store")
        return deleted_count

    except Exception as e:
        logger.error(f"Error al eliminar chunks del vector store: {e}")
        return 0


def delete_document(
    document_id: str,
    hard_delete: bool = False
) -> dict:
    """
    Elimina un documento (soft o hard delete).

    Args:
        document_id: UUID del documento
        hard_delete: Si True, elimina permanentemente. Si False, marca como 'deleted'

    Returns:
        dict: Información de la eliminación

    Raises:
        ValueError: Si el documento no existe
        Exception: Otros errores
    """
    try:
        logger.info(f"Eliminando documento {document_id} (hard={hard_delete})")

        # 1. Obtener información del documento
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
        filename = doc['filename']
        total_chunks = doc['total_chunks']

        logger.info(f"Documento encontrado: {filename} ({total_chunks} chunks)")

        # 2. Eliminar chunks del vector store
        chunks_deleted = delete_document_chunks(document_id, user_id)

        # 3. Eliminar o marcar documento
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if hard_delete:
                    # Eliminación permanente (CASCADE eliminará chunks)
                    cur.execute("""
                        DELETE FROM documents WHERE id = %s
                    """, (document_id,))
                    logger.info(f"Documento eliminado permanentemente: {document_id}")
                else:
                    # Soft delete
                    cur.execute("""
                        UPDATE documents
                        SET status = 'deleted', last_update = NOW()
                        WHERE id = %s
                    """, (document_id,))
                    logger.info(f"Documento marcado como eliminado: {document_id}")

                conn.commit()

        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "chunks_deleted": chunks_deleted,
            "hard_delete": hard_delete,
            "message": f"Documento {'eliminado' if hard_delete else 'marcado como eliminado'}: {filename}"
        }

    except Exception as e:
        logger.error(f"Error al eliminar documento {document_id}: {e}")
        raise


def delete_user_documents(
    user_id: str,
    hard_delete: bool = False
) -> dict:
    """
    Elimina todos los documentos de un usuario.

    Args:
        user_id: ID del usuario
        hard_delete: Si True, elimina permanentemente

    Returns:
        dict: Estadísticas de la eliminación
    """
    try:
        logger.info(f"Eliminando todos los documentos del usuario {user_id}")

        # Obtener lista de documentos
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, filename FROM documents
                    WHERE user_id = %s AND status != 'deleted'
                """, (user_id,))

                documents = cur.fetchall()

        if not documents:
            return {
                "status": "success",
                "deleted_count": 0,
                "message": "No hay documentos para eliminar"
            }

        # Eliminar cada documento
        deleted_count = 0
        errors = []

        for doc in documents:
            try:
                delete_document(str(doc['id']), hard_delete)
                deleted_count += 1
            except Exception as e:
                errors.append(f"{doc['filename']}: {str(e)}")
                logger.error(f"Error al eliminar {doc['filename']}: {e}")

        return {
            "status": "success" if not errors else "partial",
            "deleted_count": deleted_count,
            "total_documents": len(documents),
            "errors": errors,
            "message": f"Eliminados {deleted_count}/{len(documents)} documentos"
        }

    except Exception as e:
        logger.error(f"Error al eliminar documentos del usuario {user_id}: {e}")
        raise


def restore_document(document_id: str) -> dict:
    """
    Restaura un documento marcado como eliminado.

    Args:
        document_id: UUID del documento

    Returns:
        dict: Resultado de la restauración
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE documents
                    SET status = 'active', last_update = NOW()
                    WHERE id = %s AND status = 'deleted'
                    RETURNING filename
                """, (document_id,))

                if cur.rowcount == 0:
                    raise ValueError(f"Documento no encontrado o no está eliminado: {document_id}")

                result = cur.fetchone()
                conn.commit()

        logger.info(f"Documento restaurado: {document_id}")

        return {
            "status": "success",
            "document_id": document_id,
            "filename": result['filename'],
            "message": "Documento restaurado exitosamente"
        }

    except Exception as e:
        logger.error(f"Error al restaurar documento: {e}")
        raise


if __name__ == "__main__":
    # Test del servicio
    import sys

    if len(sys.argv) > 1:
        test_doc_id = sys.argv[1]
        hard = len(sys.argv) > 2 and sys.argv[2].lower() == "hard"

        print(f"🗑️ Eliminando documento")
        print(f"ID: {test_doc_id}")
        print(f"Tipo: {'Hard delete' if hard else 'Soft delete'}")

        result = delete_document(test_doc_id, hard_delete=hard)

        print(f"Resultado: {result}")
    else:
        print("Uso: python delete_service.py <document_id> [hard]")
