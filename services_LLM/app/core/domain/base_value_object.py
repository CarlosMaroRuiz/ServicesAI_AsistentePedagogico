"""
Base Value Object class.
Value objects are immutable and defined by their attributes, not identity.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class BaseValueObject:
    """
    Clase base para Value Objects.

    Los Value Objects son inmutables y se definen por sus atributos.
    Dos value objects con los mismos atributos son intercambiables.

    Características:
    - Inmutabilidad (frozen=True)
    - Igualdad basada en atributos
    - Sin identidad propia
    """

    def __post_init__(self):
        """Hook para validaciones personalizadas en subclases."""
        pass
