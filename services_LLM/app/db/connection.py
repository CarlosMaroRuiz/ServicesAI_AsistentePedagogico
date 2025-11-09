"""
Database connection module.
Maneja la conexión a PostgreSQL usando psycopg3 y pool de conexiones.
"""

import os
from contextlib import contextmanager
from typing import Generator
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from dotenv import load_dotenv
from loguru import logger

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL no está configurado en el archivo .env")

# Pool de conexiones global
connection_pool: ConnectionPool = None


def init_connection_pool(min_size: int = 2, max_size: int = 10) -> None:
    """
    Inicializa el pool de conexiones a PostgreSQL.

    Args:
        min_size: Número mínimo de conexiones en el pool
        max_size: Número máximo de conexiones en el pool
    """
    global connection_pool

    try:
        # Configurar keepalive para evitar que AWS RDS cierre las conexiones
        connection_pool = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=min_size,
            max_size=max_size,
            timeout=30,
            max_idle=300,  # 5 minutos de idle antes de reciclar
            reconnect_timeout=10,
            kwargs={
                "row_factory": dict_row,
                "autocommit": False,
                "prepare_threshold": None,  # Deshabilitar prepared statements
            }
        )
        logger.info(f"Pool de conexiones inicializado: min={min_size}, max={max_size}")
    except Exception as e:
        logger.error(f"Error al inicializar el pool de conexiones: {e}")
        raise


def close_connection_pool() -> None:
    """Cierra el pool de conexiones."""
    global connection_pool

    if connection_pool:
        connection_pool.close()
        logger.info("Pool de conexiones cerrado")
        connection_pool = None


@contextmanager
def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """
    Context manager para obtener una conexión del pool.

    Yields:
        psycopg.Connection: Conexión a la base de datos

    Example:
        ```python
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents")
                results = cur.fetchall()
        ```
    """
    if not connection_pool:
        init_connection_pool()

    conn = connection_pool.getconn()
    try:
        yield conn
    except Exception as e:
        conn.rollback()
        logger.error(f"Error en la transacción de base de datos: {e}")
        raise
    finally:
        connection_pool.putconn(conn)


def get_sync_connection():
    """
    Obtiene una conexión simple (sin pool) para usar con LangChain.
    Se recomienda usar get_db_connection() en su lugar cuando sea posible.

    Returns:
        psycopg.Connection: Conexión directa a la base de datos
    """
    try:
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        return conn
    except Exception as e:
        logger.error(f"Error al conectar a la base de datos: {e}")
        raise


def execute_sql_file(file_path: str) -> None:
    """
    Ejecuta un archivo SQL (útil para schema.sql).

    Args:
        file_path: Ruta al archivo SQL
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sql = f.read()
                cur.execute(sql)
                conn.commit()
                logger.info(f"Archivo SQL ejecutado correctamente: {file_path}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error al ejecutar archivo SQL {file_path}: {e}")
                raise


def test_connection() -> bool:
    """
    Prueba la conexión a la base de datos.

    Returns:
        bool: True si la conexión es exitosa
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result:
                    logger.info("Conexión a la base de datos exitosa")
                    return True
        return False
    except Exception as e:
        logger.error(f"Error al probar la conexión: {e}")
        return False


def initialize_database() -> None:
    """
    Inicializa la base de datos ejecutando el schema.sql.
    """
    import os.path

    schema_path = os.path.join(
        os.path.dirname(__file__),
        'schema.sql'
    )

    if os.path.exists(schema_path):
        execute_sql_file(schema_path)
        logger.info("Base de datos inicializada correctamente")
    else:
        logger.warning(f"Archivo schema.sql no encontrado en {schema_path}")


if __name__ == "__main__":
    # Test de conexión
    init_connection_pool()
    if test_connection():
        print("Conexión exitosa a PostgreSQL")
        initialize_database()
    else:
        print("Error al conectar a PostgreSQL")
    close_connection_pool()
