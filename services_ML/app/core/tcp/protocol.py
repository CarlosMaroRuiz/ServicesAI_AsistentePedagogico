"""
Protocolo de comunicación TCP entre services_LLM y services_ML.

Define los mensajes JSON que se intercambian vía TCP.
"""
import json
from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class TCPAction(str, Enum):
    """Acciones disponibles en el protocolo TCP."""

    # Clustering
    CLUSTER_DOCUMENTS = "cluster_documents"
    UPDATE_CLUSTERS = "update_clusters"
    GET_CLUSTERS = "get_clusters"
    GET_CLUSTER_DETAILS = "get_cluster_details"

    # Topic Modeling
    EXTRACT_TOPICS = "extract_topics"
    UPDATE_TOPICS = "update_topics"
    GET_TOPICS = "get_topics"
    GET_TOPIC_TRENDS = "get_topic_trends"

    # Recommendations
    RECOMMEND_SIMILAR = "recommend_similar"
    GET_RECOMMENDATIONS = "get_recommendations"

    # Visualization
    UPDATE_VISUALIZATION = "update_visualization"
    GET_VISUALIZATION = "get_visualization"

    # Temporal Analysis
    ANALYZE_TRENDS = "analyze_trends"
    GET_TEMPORAL_PATTERNS = "get_temporal_patterns"

    # Health Check
    PING = "ping"
    STATUS = "status"


class TCPRequest(BaseModel):
    """Estructura de solicitud TCP."""

    action: TCPAction = Field(..., description="Acción a ejecutar")
    data: Dict[str, Any] = Field(default_factory=dict, description="Datos de la solicitud")
    request_id: Optional[str] = Field(default=None, description="ID único de la solicitud")

    def to_json(self) -> str:
        """Convierte la solicitud a JSON."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "TCPRequest":
        """Crea una solicitud desde JSON."""
        return cls.model_validate_json(json_str)


class TCPResponse(BaseModel):
    """Estructura de respuesta TCP."""

    status: str = Field(..., description="Estado: success, error")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Resultado de la operación")
    error: Optional[str] = Field(default=None, description="Mensaje de error si aplica")
    request_id: Optional[str] = Field(default=None, description="ID de la solicitud original")

    def to_json(self) -> str:
        """Convierte la respuesta a JSON."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "TCPResponse":
        """Crea una respuesta desde JSON."""
        return cls.model_validate_json(json_str)

    @classmethod
    def success(
        cls, result: Dict[str, Any], request_id: Optional[str] = None
    ) -> "TCPResponse":
        """Crea una respuesta exitosa."""
        return cls(status="success", result=result, request_id=request_id)

    @classmethod
    def create_error(cls, error_message: str, request_id: Optional[str] = None) -> "TCPResponse":
        """Crea una respuesta de error."""
        return cls(status="error", error=error_message, request_id=request_id)


# ========================================
# Request Data Models
# ========================================


class ClusterDocumentsRequest(BaseModel):
    """Datos para cluster_documents."""

    user_id: str = Field(..., description="ID del usuario")
    document_ids: Optional[List[str]] = Field(
        default=None, description="IDs específicos (opcional)"
    )
    force_recluster: bool = Field(default=False, description="Forzar re-clustering")


class ExtractTopicsRequest(BaseModel):
    """Datos para extract_topics."""

    user_id: str = Field(..., description="ID del usuario")
    num_topics: Optional[int] = Field(default=None, description="Número de temas (auto si None)")
    document_ids: Optional[List[str]] = Field(default=None, description="IDs específicos")


class RecommendSimilarRequest(BaseModel):
    """Datos para recommend_similar."""

    document_id: str = Field(..., description="ID del documento de referencia")
    top_k: int = Field(default=5, description="Número de recomendaciones")
    user_id: Optional[str] = Field(default=None, description="Filtrar por usuario")


class UpdateVisualizationRequest(BaseModel):
    """Datos para update_visualization."""

    user_id: str = Field(..., description="ID del usuario")
    force_update: bool = Field(default=False, description="Forzar actualización")


class AnalyzeTrendsRequest(BaseModel):
    """Datos para analyze_trends."""

    user_id: str = Field(..., description="ID del usuario")
    window_days: int = Field(default=30, description="Ventana de análisis en días")


# ========================================
# Response Data Models
# ========================================


class ClusterInfo(BaseModel):
    """Información de un cluster."""

    cluster_id: int
    label: str
    size: int
    document_ids: List[str]
    keywords: List[str]
    centroid: Optional[List[float]] = None


class TopicInfo(BaseModel):
    """Información de un tema."""

    topic_id: int
    label: str
    keywords: List[str]
    document_count: int
    representative_docs: List[str]


class RecommendationInfo(BaseModel):
    """Información de una recomendación."""

    document_id: str
    filename: str
    similarity_score: float
    cluster_id: Optional[int] = None


class VisualizationPoint(BaseModel):
    """Punto 2D para visualización."""

    document_id: str
    x: float
    y: float
    cluster_id: int
    label: str
    filename: str


class TrendInfo(BaseModel):
    """Información de tendencia temporal."""

    topic_id: int
    topic_label: str
    trend: str  # 'increasing', 'stable', 'decreasing'
    change_percentage: float
    current_count: int
    previous_count: int


# ========================================
# Utility Functions
# ========================================


def encode_message(message: TCPRequest | TCPResponse) -> bytes:
    """
    Codifica un mensaje para envío TCP.

    Formato: [LENGTH:4bytes][JSON:Nbytes]
    """
    json_str = message.to_json()
    json_bytes = json_str.encode("utf-8")
    length = len(json_bytes)
    length_bytes = length.to_bytes(4, byteorder="big")
    return length_bytes + json_bytes


def decode_message(data: bytes) -> TCPRequest | TCPResponse:
    """
    Decodifica un mensaje recibido por TCP.

    Espera formato: [LENGTH:4bytes][JSON:Nbytes]
    """
    if len(data) < 4:
        raise ValueError("Mensaje incompleto: faltan bytes de longitud")

    length = int.from_bytes(data[:4], byteorder="big")
    json_bytes = data[4 : 4 + length]

    if len(json_bytes) < length:
        raise ValueError(f"Mensaje incompleto: esperaba {length} bytes, recibió {len(json_bytes)}")

    json_str = json_bytes.decode("utf-8")
    json_data = json.loads(json_str)

    # Detectar si es request o response
    if "action" in json_data:
        return TCPRequest.from_json(json_str)
    else:
        return TCPResponse.from_json(json_str)


def create_cluster_request(
    user_id: str, document_ids: Optional[List[str]] = None, force_recluster: bool = False
) -> TCPRequest:
    """Helper para crear una solicitud de clustering."""
    data = ClusterDocumentsRequest(
        user_id=user_id, document_ids=document_ids, force_recluster=force_recluster
    )
    return TCPRequest(action=TCPAction.CLUSTER_DOCUMENTS, data=data.model_dump())


def create_topics_request(
    user_id: str, num_topics: Optional[int] = None, document_ids: Optional[List[str]] = None
) -> TCPRequest:
    """Helper para crear una solicitud de extracción de temas."""
    data = ExtractTopicsRequest(user_id=user_id, num_topics=num_topics, document_ids=document_ids)
    return TCPRequest(action=TCPAction.EXTRACT_TOPICS, data=data.model_dump())


def create_recommendation_request(
    document_id: str, top_k: int = 5, user_id: Optional[str] = None
) -> TCPRequest:
    """Helper para crear una solicitud de recomendaciones."""
    data = RecommendSimilarRequest(document_id=document_id, top_k=top_k, user_id=user_id)
    return TCPRequest(action=TCPAction.RECOMMEND_SIMILAR, data=data.model_dump())
