"""
Base exceptions for the application.
Define custom exceptions for domain and application layers.
"""


class DomainException(Exception):
    """Base exception for all domain-related errors."""

    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code or "DOMAIN_ERROR"
        super().__init__(self.message)


class EntityNotFoundException(DomainException):
    """Exception raised when an entity is not found."""

    def __init__(self, entity_type: str, entity_id: str):
        message = f"{entity_type} with ID {entity_id} not found"
        super().__init__(message, code="ENTITY_NOT_FOUND")
        self.entity_type = entity_type
        self.entity_id = entity_id


class ValidationException(DomainException):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field: str = None):
        super().__init__(message, code="VALIDATION_ERROR")
        self.field = field


class UnauthorizedException(DomainException):
    """Exception raised when user is not authorized."""

    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(message, code="UNAUTHORIZED")


class InfrastructureException(Exception):
    """Base exception for infrastructure-related errors."""

    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code or "INFRASTRUCTURE_ERROR"
        super().__init__(self.message)


class DatabaseException(InfrastructureException):
    """Exception raised for database errors."""

    def __init__(self, message: str):
        super().__init__(message, code="DATABASE_ERROR")


class ExternalServiceException(InfrastructureException):
    """Exception raised when external service fails."""

    def __init__(self, service_name: str, message: str):
        super().__init__(
            f"{service_name} error: {message}",
            code="EXTERNAL_SERVICE_ERROR"
        )
        self.service_name = service_name
