"""
Cliente TCP para comunicación con services_ML.

Permite a services_LLM enviar comandos al servidor ML vía TCP.
"""
import socket
import asyncio
import json
from typing import Dict, Any, Optional
from loguru import logger


class MLTCPClient:
    """Cliente TCP para comunicarse con services_ML."""

    def __init__(self, host: str = "localhost", port: int = 5555, timeout: int = 30):
        """
        Inicializa el cliente TCP.

        Args:
            host: Dirección del servidor ML
            port: Puerto del servidor ML
            timeout: Timeout de conexión en segundos
        """
        self.host = host
        self.port = port
        self.timeout = timeout

    async def send_request(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Envía una solicitud al servidor ML y espera respuesta.

        Args:
            action: Acción a ejecutar (ej: "cluster_documents")
            data: Datos de la solicitud

        Returns:
            Respuesta del servidor

        Raises:
            ConnectionError: Si no puede conectarse al servidor
            TimeoutError: Si la operación excede el timeout
            ValueError: Si la respuesta tiene errores
        """
        request = {"action": action, "data": data}

        try:
            # Conectar al servidor
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=self.timeout
            )

            logger.info(f"Conectado a ML server: {self.host}:{self.port}")

            # Codificar mensaje
            json_str = json.dumps(request)
            json_bytes = json_str.encode("utf-8")
            length = len(json_bytes)
            length_bytes = length.to_bytes(4, byteorder="big")
            message = length_bytes + json_bytes

            # Enviar mensaje
            logger.debug(f"Enviando comando: {action}")
            writer.write(message)
            await writer.drain()

            # Leer respuesta (longitud)
            response_length_bytes = await asyncio.wait_for(
                reader.readexactly(4), timeout=self.timeout
            )
            response_length = int.from_bytes(response_length_bytes, byteorder="big")

            # Leer respuesta (contenido)
            response_bytes = await asyncio.wait_for(
                reader.readexactly(response_length), timeout=self.timeout
            )
            response_str = response_bytes.decode("utf-8")
            response = json.loads(response_str)

            logger.debug(f"Respuesta recibida: {response.get('status')}")

            # Cerrar conexión
            writer.close()
            await writer.wait_closed()

            # Verificar errores
            if response.get("status") == "error":
                raise ValueError(f"Error del servidor ML: {response.get('error')}")

            return response.get("result", {})

        except asyncio.TimeoutError:
            logger.error(f"Timeout conectando a ML server ({self.timeout}s)")
            raise TimeoutError(f"Timeout conectando a ML server después de {self.timeout}s")

        except ConnectionRefusedError:
            logger.error(f"No se puede conectar a ML server en {self.host}:{self.port}")
            raise ConnectionError(f"ML server no disponible en {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Error en comunicación TCP: {str(e)}", exc_info=True)
            raise

    # ========================================
    # Métodos de conveniencia
    # ========================================

    async def cluster_documents(
        self, user_id: str, document_ids: Optional[list] = None, force_recluster: bool = False
    ) -> Dict[str, Any]:
        """
        Solicita clustering de documentos.

        Args:
            user_id: ID del usuario
            document_ids: IDs específicos (opcional)
            force_recluster: Forzar re-clustering

        Returns:
            Resultado con clusters encontrados
        """
        data = {
            "user_id": user_id,
            "document_ids": document_ids,
            "force_recluster": force_recluster,
        }
        return await self.send_request("cluster_documents", data)

    async def extract_topics(
        self, user_id: str, num_topics: Optional[int] = None, document_ids: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Solicita extracción de temas.

        Args:
            user_id: ID del usuario
            num_topics: Número de temas (auto si None)
            document_ids: IDs específicos (opcional)

        Returns:
            Resultado con temas extraídos
        """
        data = {"user_id": user_id, "num_topics": num_topics, "document_ids": document_ids}
        return await self.send_request("extract_topics", data)

    async def recommend_similar(
        self, document_id: str, top_k: int = 5, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solicita recomendaciones de documentos similares.

        Args:
            document_id: ID del documento de referencia
            top_k: Número de recomendaciones
            user_id: Filtrar por usuario (opcional)

        Returns:
            Lista de documentos similares
        """
        data = {"document_id": document_id, "top_k": top_k, "user_id": user_id}
        return await self.send_request("recommend_similar", data)

    async def update_visualization(
        self, user_id: str, force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Solicita actualización de visualización 2D.

        Args:
            user_id: ID del usuario
            force_update: Forzar actualización

        Returns:
            Puntos 2D para visualización
        """
        data = {"user_id": user_id, "force_update": force_update}
        return await self.send_request("update_visualization", data)

    async def analyze_trends(self, user_id: str, window_days: int = 30) -> Dict[str, Any]:
        """
        Solicita análisis de tendencias temporales.

        Args:
            user_id: ID del usuario
            window_days: Ventana de análisis en días

        Returns:
            Tendencias y patrones temporales
        """
        data = {"user_id": user_id, "window_days": window_days}
        return await self.send_request("analyze_trends", data)

    async def ping(self) -> Dict[str, Any]:
        """
        Verifica si el servidor ML está disponible.

        Returns:
            Mensaje de confirmación
        """
        return await self.send_request("ping", {})

    async def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado del servidor ML.

        Returns:
            Información de estado
        """
        return await self.send_request("status", {})


# ========================================
# Factory Function
# ========================================


def create_ml_client(host: str = "localhost", port: int = 5555, timeout: int = 30) -> MLTCPClient:
    """
    Crea una instancia del cliente ML TCP.

    Args:
        host: Dirección del servidor ML
        port: Puerto del servidor ML
        timeout: Timeout en segundos

    Returns:
        Cliente TCP configurado
    """
    return MLTCPClient(host=host, port=port, timeout=timeout)
