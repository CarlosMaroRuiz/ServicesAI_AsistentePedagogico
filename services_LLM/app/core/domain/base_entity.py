"""
Base Entity class for all domain entities.
Provides common functionality like ID generation and timestamps.
"""
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


class BaseEntity:
    """
    Clase base para todas las entidades del dominio.

    Una entidad se define por su identidad (ID), no por sus atributos.
    Dos entidades con el mismo ID son la misma entidad, incluso si sus
    atributos difieren.
    """

    def __init__(
        self,
        entity_id: Optional[UUID] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        self.id = entity_id or uuid4()
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()

    def __eq__(self, other) -> bool:
        """Dos entidades son iguales si tienen el mismo ID."""
        if not isinstance(other, BaseEntity):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash basado en el ID de la entidad."""
        return hash(self.id)

    def _update_timestamp(self) -> None:
        """Actualiza el timestamp de última modificación."""
        self.updated_at = datetime.now()
