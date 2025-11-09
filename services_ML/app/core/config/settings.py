"""
Configuración global del servicio ML.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Settings del servicio ML."""

    # Service Configuration
    ml_service_host: str = Field(default="0.0.0.0", env="ML_SERVICE_HOST")
    ml_service_port: int = Field(default=8001, env="ML_SERVICE_PORT")

    # TCP Server Configuration
    tcp_server_host: str = Field(default="0.0.0.0", env="TCP_SERVER_HOST")
    tcp_server_port: int = Field(default=5555, env="TCP_SERVER_PORT")

    # Database Configuration
    db_host: str = Field(env="DB_HOST")
    db_port: int = Field(default=5432, env="DB_PORT")
    db_name: str = Field(env="DB_NAME")
    db_user: str = Field(env="DB_USER")
    db_password: str = Field(env="DB_PASSWORD")
    database_url: str = Field(env="DATABASE_URL")

    # HDBSCAN Configuration
    min_cluster_size: int = Field(default=3, env="MIN_CLUSTER_SIZE")
    min_samples: int = Field(default=2, env="MIN_SAMPLES")
    cluster_selection_method: str = Field(default="eom", env="CLUSTER_SELECTION_METHOD")

    # UMAP Configuration
    umap_n_neighbors: int = Field(default=15, env="UMAP_N_NEIGHBORS")
    umap_n_components_cluster: int = Field(default=5, env="UMAP_N_COMPONENTS_CLUSTER")
    umap_n_components_viz: int = Field(default=2, env="UMAP_N_COMPONENTS_VIZ")
    umap_metric: str = Field(default="cosine", env="UMAP_METRIC")
    umap_min_dist: float = Field(default=0.1, env="UMAP_MIN_DIST")

    # BERTopic Configuration
    top_n_words: int = Field(default=10, env="TOP_N_WORDS")
    topic_language: str = Field(default="spanish", env="TOPIC_LANGUAGE")
    calculate_probabilities: bool = Field(default=False, env="CALCULATE_PROBABILITIES")

    # Recommendations Configuration
    top_k_recommendations: int = Field(default=5, env="TOP_K_RECOMMENDATIONS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Temporal Analysis Configuration
    analysis_window_days: int = Field(default=30, env="ANALYSIS_WINDOW_DAYS")
    trend_threshold: float = Field(default=0.2, env="TREND_THRESHOLD")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file_retention: str = Field(default="30 days", env="LOG_FILE_RETENTION")

    # Application Settings
    max_documents_per_batch: int = Field(default=1000, env="MAX_DOCUMENTS_PER_BATCH")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
settings = Settings()
