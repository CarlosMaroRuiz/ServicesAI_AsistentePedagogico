
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger

from app.services.chat_service import (
    chat_with_rag,
    chat_with_rag_stream,
    create_or_get_session,
    get_user_sessions,
    get_chat_history_from_db
)
from app.models.chat_model import (
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatHistoryResponse
)
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Realiza una consulta de chat con RAG (sin streaming).

    - **user_id**: ID del usuario
    - **message**: Pregunta/mensaje del usuario
    - **session_id**: ID de sesión (opcional, se crea uno si no se proporciona)
    - **use_history**: Si usar historial de conversación (default: True)
    - **max_history**: Máximo de mensajes de historial a usar (default: 5)
    - **top_k**: Número de chunks relevantes a recuperar (default: 3)

    Returns:
        ChatResponse con la respuesta, fuentes, session_id, etc.
    """
    try:
        logger.info(f"Chat request: user={request.user_id}, message='{request.message[:50]}...'")

        result = chat_with_rag(
            user_id=request.user_id,
            question=request.message,
            session_id=request.session_id,
            use_history=request.use_history,
            max_history=request.max_history,
            top_k=request.top_k
        )

        return ChatResponse(
            answer=result['answer'],
            sources=result['sources'],
            session_id=result['session_id'],
            tokens_used=result.get('tokens_used'),
            model=result.get('model', 'deepseek-chat')
        )

    except Exception as e:
        logger.error(f"Error en chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el chat: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Realiza una consulta de chat con RAG usando Server-Sent Events (SSE) para streaming en tiempo real.

    Este endpoint retorna la respuesta del asistente palabra por palabra conforme se genera,
    permitiendo una experiencia de usuario más fluida.

    - **user_id**: ID del usuario
    - **message**: Pregunta/mensaje del usuario
    - **session_id**: ID de sesión (opcional, se crea uno si no se proporciona)
    - **use_history**: Si usar historial de conversación (default: True)
    - **max_history**: Máximo de mensajes de historial a usar (default: 5)
    - **top_k**: Número de chunks relevantes a recuperar (default: 3)

    Returns:
        StreamingResponse con eventos SSE en el siguiente formato:
        - `data: {"type": "sources", "content": [...]}` - Fuentes encontradas
        - `data: {"type": "content", "content": "texto"}` - Chunks de la respuesta
        - `data: {"type": "done", "session_id": "..."}` - Finalización
        - `data: {"type": "error", "content": "..."}` - Error si ocurre

    Ejemplo de uso con JavaScript:
    ```javascript
    const eventSource = new EventSource('/chat/stream?user_id=123&message=hola');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'content') {
            console.log(data.content);
        }
    };
    ```
    """
    try:
        logger.info(f"Chat streaming request: user={request.user_id}, message='{request.message[:50]}...'")

        return StreamingResponse(
            chat_with_rag_stream(
                user_id=request.user_id,
                question=request.message,
                session_id=request.session_id,
                use_history=request.use_history,
                max_history=request.max_history,
                top_k=request.top_k
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )

    except Exception as e:
        logger.error(f"Error en chat streaming: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el chat: {str(e)}"
        )


@router.post("/session", response_model=ChatSessionResponse)
async def create_session(request: ChatSessionCreate):
    """
    Crea una nueva sesión de chat.

    - **user_id**: ID del usuario
    - **title**: Título de la sesión (opcional)
    """
    try:
        from app.db.connection import get_db_connection

        session_id = create_or_get_session(
            user_id=request.user_id,
            title=request.title
        )

        # Recuperar la sesión completa de la base de datos
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        id, session_id, user_id, title,
                        created_at, last_activity
                    FROM chat_sessions
                    WHERE session_id = %s
                """, (session_id,))

                session = cur.fetchone()

        if session:
            return ChatSessionResponse(
                id=session['id'],
                session_id=session['session_id'],
                user_id=session['user_id'],
                title=session['title'],
                created_at=session['created_at'],
                last_activity=session['last_activity'],
                message_count=0
            )
        else:
            raise HTTPException(status_code=500, detail="Sesión creada pero no encontrada")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al crear sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{user_id}")
async def list_sessions(
    user_id: str,
    limit: int = Query(10, ge=1, le=100)
):
    """
    Lista las sesiones de chat de un usuario.

    - **user_id**: ID del usuario
    - **limit**: Número máximo de sesiones a retornar
    """
    try:
        sessions = get_user_sessions(user_id, limit)

        return {
            "total": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        logger.error(f"Error al listar sesiones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}", response_model=ChatHistoryResponse)
async def get_history(
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Obtiene el historial de chat de un usuario.

    - **user_id**: ID del usuario
    - **session_id**: ID de sesión específica (opcional)
    - **limit**: Número máximo de mensajes
    """
    try:
        from app.db.connection import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute("""
                        SELECT
                            id, user_id, session_id, message, response,
                            sources, created_at, metadata
                        FROM chat_history
                        WHERE user_id = %s AND session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, session_id, limit))
                else:
                    cur.execute("""
                        SELECT
                            id, user_id, session_id, message, response,
                            sources, created_at, metadata
                        FROM chat_history
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, limit))

                history = cur.fetchall()

        return ChatHistoryResponse(
            total=len(history),
            history=history,
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error al obtener historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{user_id}")
async def clear_history(
    user_id: str,
    session_id: Optional[str] = Query(None)
):
    """
    Elimina el historial de chat.

    - **user_id**: ID del usuario
    - **session_id**: ID de sesión específica (opcional, si no se proporciona elimina todo)
    """
    try:
        from app.db.connection import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute("""
                        DELETE FROM chat_history
                        WHERE user_id = %s AND session_id = %s
                    """, (user_id, session_id))
                else:
                    cur.execute("""
                        DELETE FROM chat_history
                        WHERE user_id = %s
                    """, (user_id,))

                deleted_count = cur.rowcount
                conn.commit()

        return {
            "success": True,
            "message": f"Eliminados {deleted_count} mensajes",
            "deleted_count": deleted_count
        }

    except Exception as e:
        logger.error(f"Error al eliminar historial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200)
):
    """
    Obtiene todos los mensajes de una sesión específica.

    - **session_id**: ID de la sesión
    - **limit**: Número máximo de mensajes
    """
    try:
        from app.db.connection import get_db_connection

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        id, message, response, sources, created_at
                    FROM chat_history
                    WHERE session_id = %s
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (session_id, limit))

                messages = cur.fetchall()

        return {
            "session_id": session_id,
            "total": len(messages),
            "messages": messages
        }

    except Exception as e:
        logger.error(f"Error al obtener mensajes de sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))
