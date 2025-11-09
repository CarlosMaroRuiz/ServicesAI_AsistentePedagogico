"""
Conexión a PostgreSQL para services_ML.

Reutiliza la misma base de datos que services_LLM.
"""
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from loguru import logger

from app.core.config import settings


# Pool de conexiones global
connection_pool: ConnectionPool | None = None


def init_connection_pool() -> ConnectionPool:
    """
    Inicializa el pool de conexiones a PostgreSQL.

    Returns:
        Pool de conexiones
    """
    global connection_pool

    try:
        logger.info("Inicializando pool de conexiones a PostgreSQL...")

        # Configurar keepalive para evitar que AWS RDS cierre las conexiones
        connection_pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=2,
            max_size=10,
            open=True,
            timeout=30,
            max_idle=300,  # 5 minutos de idle antes de reciclar
            reconnect_timeout=10,
            kwargs={
                "row_factory": dict_row,
                "autocommit": False,
                "prepare_threshold": None,  # Deshabilitar prepared statements
            }
        )

        logger.info("✅ Pool de conexiones inicializado")
        return connection_pool

    except Exception as e:
        logger.error(f"❌ Error inicializando pool de conexiones: {str(e)}", exc_info=True)
        raise


def get_connection():
    """
    Obtiene una conexión del pool.

    Returns:
        Conexión de PostgreSQL

    Raises:
        RuntimeError: Si el pool no está inicializado
    """
    global connection_pool

    if connection_pool is None:
        raise RuntimeError("Pool de conexiones no inicializado. Llamar a init_connection_pool() primero")

    return connection_pool.connection()


def close_connection_pool():
    """Cierra el pool de conexiones."""
    global connection_pool

    if connection_pool:
        logger.info("Cerrando pool de conexiones...")
        connection_pool.close()
        connection_pool = None
        logger.info("✅ Pool de conexiones cerrado")


def test_connection() -> bool:
    """
    Prueba la conexión a la base de datos.

    Returns:
        True si la conexión es exitosa
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                logger.info("✅ Conexión a PostgreSQL exitosa")
                return result is not None
    except Exception as e:
        logger.error(f"❌ Error probando conexión: {str(e)}", exc_info=True)
        return False
