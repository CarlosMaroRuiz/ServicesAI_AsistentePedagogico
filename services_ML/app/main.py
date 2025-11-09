"""
Main entry point para services_ML.

Inicia tanto el servidor FastAPI (REST) como el servidor TCP.
"""
import sys
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.core.tcp import create_tcp_server
from app.core.tcp.handlers import register_tcp_handlers
from app.core.database import init_connection_pool, close_connection_pool, test_connection

# Configurar logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level,
)
logger.add(
    "logs/ml_service_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention=settings.log_file_retention,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
)


# Referencia global al servidor TCP
tcp_server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja el ciclo de vida de la aplicación.

    Inicia el servidor TCP al arrancar y lo detiene al cerrar.
    """
    global tcp_server

    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Iniciando services_ML...")
    logger.info("=" * 60)

    try:
        # Inicializar pool de conexiones
        init_connection_pool()
        if test_connection():
            logger.info("✅ Conexión a PostgreSQL establecida")
        else:
            logger.error("❌ No se pudo conectar a PostgreSQL")

        # Crear e iniciar servidor TCP en background
        tcp_server = create_tcp_server()

        # Registrar handlers TCP
        register_tcp_handlers(tcp_server)

        # Iniciar servidor TCP
        await tcp_server.start_in_background()

        logger.info(f"✅ FastAPI escuchando en {settings.ml_service_host}:{settings.ml_service_port}")
        logger.info(f"✅ Servidor TCP escuchando en {settings.tcp_server_host}:{settings.tcp_server_port}")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"❌ Error durante el inicio: {str(e)}", exc_info=True)
        raise

    # Shutdown
    logger.info("=" * 60)
    logger.info("⏹️  Deteniendo services_ML...")
    logger.info("=" * 60)

    try:
        if tcp_server:
            await tcp_server.stop()

        # Cerrar pool de conexiones
        close_connection_pool()

        logger.info("✅ Servicio detenido correctamente")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Error durante el apagado: {str(e)}", exc_info=True)


# Crear aplicación FastAPI
app = FastAPI(
    title="ML Service API",
    description="Servicio de Machine Learning No Supervisado para análisis de recursos educativos",
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


# ========================================
# Documentación
# ========================================


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


# ========================================
# Endpoints de Health Check
# ========================================


@app.get("/health")
async def health_check():
    """Verifica que el servicio esté funcionando."""
    return {
        "status": "healthy",
        "service": "services_ML",
        "version": "1.0.0",
        "tcp_server": {
            "host": settings.tcp_server_host,
            "port": settings.tcp_server_port,
            "status": "running" if tcp_server and tcp_server.server else "stopped",
        },
    }


@app.get("/info")
async def service_info():
    """Información del servicio."""
    return {
        "name": "ML Service",
        "version": "1.0.0",
        "description": "Machine Learning No Supervisado para análisis educativo",
        "features": [
            "Clustering automático (HDBSCAN)",
            "Topic modeling (BERTopic)",
            "Recomendaciones (KNN + Cosine Similarity)",
            "Visualización 2D (UMAP)",
            "Análisis temporal de tendencias",
        ],
        "models": {
            "clustering": "HDBSCAN",
            "dimensionality_reduction": "UMAP",
            "topic_modeling": "BERTopic",
            "recommendations": "KNN + Cosine Similarity",
        },
        "configuration": {
            "min_cluster_size": settings.min_cluster_size,
            "min_samples": settings.min_samples,
            "umap_n_neighbors": settings.umap_n_neighbors,
            "top_k_recommendations": settings.top_k_recommendations,
        },
    }


@app.get("/")
async def root():
    """Endpoint raíz."""
    return {
        "message": "ML Service API",
        "docs": "/docs",
        "health": "/health",
        "info": "/info",
    }


# ========================================
# Incluir routers de features
# ========================================

from app.features.clustering.infrastructure.http import controllers as clustering_controllers
from app.features.topic_modeling.infrastructure.http import controllers as topics_controllers
from app.features.recommendations.infrastructure.http import controllers as recommendations_controllers
from app.features.visualization.infrastructure.http import controllers as visualization_controllers

app.include_router(clustering_controllers.router, prefix="/clustering", tags=["clustering"])
app.include_router(topics_controllers.router, prefix="/topics", tags=["topics"])
app.include_router(recommendations_controllers.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(visualization_controllers.router, prefix="/visualization", tags=["visualization"])


# ========================================
# Entry Point
# ========================================

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Iniciando servidor uvicorn...")

    uvicorn.run(
        "app.main:app",
        host=settings.ml_service_host,
        port=settings.ml_service_port,
        reload=False,  # Cambiar a True para desarrollo
        log_level=settings.log_level.lower(),
    )
