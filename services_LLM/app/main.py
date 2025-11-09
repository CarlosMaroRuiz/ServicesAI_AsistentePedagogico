import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
import sys

from app.routes import document_routes, chat_routes, ml_routes
from app.db.connection import (
    init_connection_pool,
    close_connection_pool,
    test_connection,
    initialize_database
)
from app.models.response_model import HealthCheckResponse, ErrorResponse

# Configurar logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
)

# Cargar variables de entorno
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja el ciclo de vida de la aplicación.
    Inicializa conexiones al inicio y las cierra al finalizar.
    """
    # Startup
    logger.info("Iniciando aplicación...")

    try:
        # Crear directorio de logs
        os.makedirs("logs", exist_ok=True)

        # Inicializar pool de conexiones
        logger.info("📦 Inicializando pool de conexiones a PostgreSQL...")
        init_connection_pool(min_size=2, max_size=10)

        # Test de conexión
        if test_connection():
            logger.info("Conexión a PostgreSQL exitosa")
        else:
            logger.error("Error al conectar a PostgreSQL")
            raise Exception("No se pudo conectar a la base de datos")

        # Inicializar base de datos (ejecutar schema.sql si es necesario)
        logger.info("Inicializando esquema de base de datos...")
        try:
            initialize_database()
            logger.info("Base de datos inicializada")
        except Exception as e:
            logger.warning(f"Error al inicializar DB (puede ser que ya exista): {e}")

        logger.info("Aplicación iniciada correctamente")

    except Exception as e:
        logger.error(f"Error durante startup: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Cerrando aplicación...")
    close_connection_pool()
    logger.info("👋 Aplicación cerrada")


# Crear aplicación FastAPI
app = FastAPI(
    title="RAG Conversacional API",
    description="Sistema de chat conversacional con RAG usando DeepSeek, LangChain y PostgreSQL (pgvector)",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,  # Deshabilitamos Swagger UI default
    redoc_url=None  # Deshabilitamos ReDoc default
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para loggear todas las requests."""
    logger.info(f"📨 {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"📤 {request.method} {request.url.path} - Status: {response.status_code}")
    return response


# Exception handler global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global para excepciones no manejadas."""
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Error interno del servidor",
            detail=str(exc)
        ).dict()
    )


# Incluir routers
app.include_router(document_routes.router)
app.include_router(chat_routes.router)
app.include_router(ml_routes.router)


# Rutas de documentación
@app.get("/docs", include_in_schema=False)
async def scalar_html():
    """Documentación API con Scalar."""
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
    <title>{app.title}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
    <script
        id="api-reference"
        data-url="/openapi.json"
        data-configuration='{{"theme": "purple"}}'></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>
""")


# Rutas básicas
@app.get("/", tags=["root"])
async def root():
    """Endpoint raíz."""
    return {
        "message": "RAG Conversacional API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "documents": "/documents",
            "chat": "/chat",
            "ml": "/ml",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.
    Verifica el estado de la aplicación y sus dependencias.
    """
    from datetime import datetime

    # Test conexión a DB
    db_healthy = test_connection()

    # Test vector store (simplificado)
    vector_store_healthy = db_healthy  # Si DB funciona, vector store también

    status = "healthy" if (db_healthy and vector_store_healthy) else "unhealthy"

    return HealthCheckResponse(
        status=status,
        database=db_healthy,
        vector_store=vector_store_healthy,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/info", tags=["info"])
async def get_info():
    """
    Obtiene información sobre la configuración del sistema.
    """
    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL", "N/A"),
        "llm_model": os.getenv("DEEPSEEK_MODEL", "N/A"),
        "database": "PostgreSQL + pgvector",
        "framework": "LangChain + FastAPI",
        "max_file_size_mb": os.getenv("MAX_FILE_SIZE_MB", "50"),
        "chunk_size": os.getenv("CHUNK_SIZE", "1000"),
        "chunk_overlap": os.getenv("CHUNK_OVERLAP", "200")
    }


if __name__ == "__main__":
    import uvicorn

    # Ejecutar servidor
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )
