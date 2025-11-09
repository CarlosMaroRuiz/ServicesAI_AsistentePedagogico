"""
Servidor TCP para comunicación entre services_LLM y services_ML.

Escucha conexiones en el puerto configurado y procesa comandos JSON.
"""
import asyncio
import socket
from typing import Callable, Dict, Any
from loguru import logger

from app.core.config import settings
from app.core.tcp.protocol import (
    TCPRequest,
    TCPResponse,
    TCPAction,
    encode_message,
    decode_message,
)


class TCPServer:
    """Servidor TCP asíncrono."""

    def __init__(self, host: str, port: int):
        """
        Inicializa el servidor TCP.

        Args:
            host: Dirección IP para escuchar
            port: Puerto para escuchar
        """
        self.host = host
        self.port = port
        self.handlers: Dict[TCPAction, Callable] = {}
        self.server: asyncio.Server | None = None

    def register_handler(self, action: TCPAction, handler: Callable):
        """
        Registra un handler para una acción específica.

        Args:
            action: Acción TCP
            handler: Función async que procesa la acción
        """
        self.handlers[action] = handler
        logger.debug(f"Handler registrado: {action.value} -> {handler.__name__}")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Maneja una conexión de cliente.

        Args:
            reader: Stream de lectura
            writer: Stream de escritura
        """
        addr = writer.get_extra_info("peername")
        logger.info(f"Conexión TCP recibida desde {addr}")

        try:
            # Leer longitud del mensaje (4 bytes)
            length_bytes = await reader.readexactly(4)
            message_length = int.from_bytes(length_bytes, byteorder="big")

            logger.debug(f"Esperando mensaje de {message_length} bytes")

            # Leer mensaje completo
            message_bytes = await reader.readexactly(message_length)
            full_message = length_bytes + message_bytes

            # Decodificar mensaje
            request = decode_message(full_message)

            if not isinstance(request, TCPRequest):
                raise ValueError("Mensaje recibido no es una solicitud válida")

            logger.info(f"Solicitud recibida: {request.action.value}")
            logger.debug(f"Datos: {request.data}")

            # Procesar solicitud
            response = await self.process_request(request)

            # Enviar respuesta
            response_bytes = encode_message(response)
            writer.write(response_bytes)
            await writer.drain()

            logger.info(f"Respuesta enviada: {response.status}")

        except asyncio.IncompleteReadError:
            logger.error("Conexión cerrada inesperadamente por el cliente")
            response = TCPResponse.create_error("Conexión cerrada inesperadamente")
            writer.write(encode_message(response))
            await writer.drain()

        except Exception as e:
            logger.error(f"Error procesando solicitud: {str(e)}", exc_info=True)
            response = TCPResponse.create_error(f"Error del servidor: {str(e)}")
            writer.write(encode_message(response))
            await writer.drain()

        finally:
            logger.info(f"Cerrando conexión con {addr}")
            writer.close()
            await writer.wait_closed()

    async def process_request(self, request: TCPRequest) -> TCPResponse:
        """
        Procesa una solicitud TCP.

        Args:
            request: Solicitud recibida

        Returns:
            Respuesta TCP
        """
        try:
            # Buscar handler
            handler = self.handlers.get(request.action)

            if handler is None:
                return TCPResponse.create_error(
                    f"Acción no soportada: {request.action.value}", request_id=request.request_id
                )

            # Ejecutar handler
            result = await handler(request.data)

            # Retornar respuesta exitosa
            return TCPResponse.success(result, request_id=request.request_id)

        except Exception as e:
            logger.error(f"Error en handler {request.action.value}: {str(e)}", exc_info=True)
            return TCPResponse.create_error(str(e), request_id=request.request_id)

    async def start(self):
        """Inicia el servidor TCP."""
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)

        addr = self.server.sockets[0].getsockname() if self.server.sockets else ("", 0)
        logger.info(f"✅ Servidor TCP iniciado en {addr[0]}:{addr[1]}")

        async with self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Detiene el servidor TCP."""
        if self.server:
            logger.info("Deteniendo servidor TCP...")
            self.server.close()
            await self.server.wait_closed()
            logger.info("✅ Servidor TCP detenido")

    async def start_in_background(self):
        """Inicia el servidor en segundo plano."""
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)

        addr = self.server.sockets[0].getsockname() if self.server.sockets else ("", 0)
        logger.info(f"✅ Servidor TCP iniciado en {addr[0]}:{addr[1]}")

        # Iniciar en background sin bloquear
        asyncio.create_task(self.server.serve_forever())


# ========================================
# Handlers de Ejemplo (temporales)
# ========================================


async def handle_ping(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handler de ejemplo para PING."""
    return {"message": "pong", "status": "healthy"}


async def handle_status(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handler de ejemplo para STATUS."""
    return {
        "service": "services_ML",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "clustering",
            "topic_modeling",
            "recommendations",
            "visualization",
            "temporal_analysis",
        ],
    }


# ========================================
# Factory Function
# ========================================


def create_tcp_server() -> TCPServer:
    """
    Crea una instancia del servidor TCP con handlers registrados.

    Returns:
        Servidor TCP configurado
    """
    server = TCPServer(settings.tcp_server_host, settings.tcp_server_port)

    # Registrar handlers básicos
    server.register_handler(TCPAction.PING, handle_ping)
    server.register_handler(TCPAction.STATUS, handle_status)

    # Los demás handlers se registrarán desde main.py cuando se importen
    # los módulos de features (clustering, topics, etc.)

    return server
