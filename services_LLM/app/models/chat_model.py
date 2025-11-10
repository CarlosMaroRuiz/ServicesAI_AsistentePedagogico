"""
Modelos Pydantic para chat y conversaciones.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Mensaje individual en el chat."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """Request para el endpoint de chat."""
    user_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=10000)
    session_id: Optional[str] = None
    use_history: bool = True
    max_history: int = Field(default=5, ge=0, le=20)
    top_k: int = Field(default=3, ge=1, le=10)

    model_config = {"json_schema_extra": {
        "example": {
            "user_id": "user123",
            "message": "¿Qué dice el contrato sobre los pagos?",
            "session_id": "session-abc-123",
            "use_history": True,
            "max_history": 5,
            "top_k": 3
        }
    }}


class Source(BaseModel):
    """Fuente de información para la respuesta."""
    content: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    chunk_index: Optional[int] = None
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    """Respuesta del chat."""
    answer: str
    sources: List[Source] = []
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: Optional[int] = None
    model: str = "deepseek-chat"

    model_config = {"json_schema_extra": {
        "example": {
            "answer": "Según el contrato, los pagos se realizan mensualmente...",
            "sources": [
                {
                    "content": "Cláusula 5: Los pagos se efectuarán...",
                    "document_id": "uuid-123",
                    "filename": "contrato.pdf",
                    "chunk_index": 12
                }
            ],
            "session_id": "session-abc-123",
            "timestamp": "2025-01-06T10:30:00",
            "model": "deepseek-chat"
        }
    }}


class ChatHistoryItem(BaseModel):
    """Item del historial de chat."""
    id: int
    user_id: str
    session_id: Optional[str]
    message: str
    response: str
    sources: List[dict] = []
    created_at: datetime
    metadata: dict = {}


class ChatHistoryResponse(BaseModel):
    """Respuesta con historial de chat."""
    total: int
    history: List[ChatHistoryItem]
    session_id: Optional[str] = None


class ChatSessionCreate(BaseModel):
    """Crear nueva sesión de chat."""
    user_id: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    """Respuesta de sesión de chat."""
    id: int
    session_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
