from typing import Optional, Any
from pydantic import BaseModel, Field


class SuccessResponse(BaseModel):
    """Respuesta exitosa genérica."""
    success: bool = True
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Respuesta de error."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Respuesta del health check."""
    status: str = "healthy"
    database: bool
    vector_store: bool
    api_version: str = "1.0.0"
    timestamp: str


class UploadResponse(BaseModel):
    """Respuesta al subir un archivo."""
    success: bool = True
    message: str
    document_id: str
    filename: str
    chunks_created: int
    file_size_mb: float


class BatchOperationResponse(BaseModel):
    """Respuesta para operaciones por lote."""
    total: int
    successful: int
    failed: int
    errors: list[str] = []
    results: list[Any] = []


class PaginationMeta(BaseModel):
    """Metadata de paginación."""
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_next: bool
    has_previous: bool


class PaginatedResponse(BaseModel):
    """Respuesta paginada genérica."""
    items: list[Any]
    meta: PaginationMeta
