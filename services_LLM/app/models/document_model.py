"""
Modelos Pydantic para documentos.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


class DocumentBase(BaseModel):
    """Modelo base para documentos."""
    user_id: str = Field(..., min_length=1, max_length=100)
    filename: str = Field(..., min_length=1)


class DocumentCreate(DocumentBase):
    """Modelo para crear un documento."""
    file_size_bytes: Optional[int] = None
    metadata: Optional[dict] = None


class DocumentUpdate(BaseModel):
    """Modelo para actualizar un documento."""
    filename: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(active|inactive|deleted)$")
    metadata: Optional[dict] = None


class DocumentResponse(DocumentBase):
    """Modelo de respuesta para documentos."""
    id: UUID
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    upload_date: datetime
    last_update: datetime
    total_chunks: int = 0
    status: str = "active"
    metadata: dict = {}

    model_config = ConfigDict(from_attributes=True)


class DocumentStats(BaseModel):
    """Estadísticas de un documento."""
    id: UUID
    filename: str
    total_chunks: int
    messages_count: int = 0
    upload_date: datetime
    last_update: datetime


class DocumentListResponse(BaseModel):
    """Respuesta para lista de documentos."""
    total: int
    documents: list[DocumentResponse]
    page: int = 1
    page_size: int = 10


class DocumentDeleteResponse(BaseModel):
    """Respuesta al eliminar un documento."""
    success: bool
    message: str
    document_id: UUID
    chunks_deleted: int = 0


class PedagogicalContentResponse(BaseModel):
    """Respuesta de contenido pedagógico extraído."""
    success: bool
    document_id: str
    filename: str
    pedagogical_content: dict
    message: str


class PedagogicalSearchResult(BaseModel):
    """Resultado de búsqueda pedagógica."""
    document_id: str
    filename: str
    content: list


class PedagogicalSearchResponse(BaseModel):
    """Respuesta de búsqueda pedagógica."""
    success: bool
    query_type: str
    total_results: int
    results: list[PedagogicalSearchResult]
    message: str
