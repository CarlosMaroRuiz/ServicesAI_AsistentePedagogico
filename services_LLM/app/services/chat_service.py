"""
Servicio de chat conversacional con RAG.
Usa DeepSeek API y LangChain moderno (create_retrieval_chain).
"""

import os
import uuid
from typing import List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
from loguru import logger
from psycopg.types.json import Json

from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.services.query_service import get_relevant_chunks_with_scores, create_retriever
from app.db.connection import get_db_connection

# Cargar variables de entorno
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_deepseek_client() -> OpenAI:
    """
    Obtiene el cliente de DeepSeek API.

    Returns:
        OpenAI: Cliente configurado para DeepSeek
    """
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )


def get_chat_history_from_db(
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 5
) -> List[Tuple[str, str]]:
    """
    Obtiene el historial de chat desde la base de datos.

    Args:
        user_id: ID del usuario
        session_id: ID de la sesión (opcional)
        limit: Número máximo de mensajes a recuperar

    Returns:
        List[Tuple[str, str]]: Lista de (pregunta, respuesta)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute("""
                        SELECT message, response
                        FROM chat_history
                        WHERE user_id = %s AND session_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, session_id, limit))
                else:
                    cur.execute("""
                        SELECT message, response
                        FROM chat_history
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (user_id, limit))

                results = cur.fetchall()
                # Invertir para que estén en orden cronológico
                history = [(r['message'], r['response']) for r in reversed(results)]
                return history

    except Exception as e:
        logger.error(f"Error al obtener historial de chat: {e}")
        return []


def save_chat_to_db(
    user_id: str,
    message: str,
    response: str,
    session_id: str,
    sources: List[dict]
) -> None:
    """
    Guarda un intercambio de chat en la base de datos.

    Args:
        user_id: ID del usuario
        message: Mensaje del usuario
        response: Respuesta del asistente
        session_id: ID de la sesión
        sources: Fuentes usadas para la respuesta
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_history (user_id, session_id, message, response, sources)
                    VALUES (%s, %s, %s, %s, %s)
                """, (user_id, session_id, message, response, Json(sources)))
                conn.commit()

        logger.info(f"Chat guardado en DB: user={user_id}, session={session_id}")

    except Exception as e:
        logger.error(f"Error al guardar chat en DB: {e}")


def format_chat_history_for_prompt(
    history: List[Tuple[str, str]]
) -> List:
    """
    Formatea el historial de chat para el prompt de LangChain.

    Args:
        history: Lista de (pregunta, respuesta)

    Returns:
        List: Lista de mensajes formateados
    """
    messages = []
    for question, answer in history:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


def chat_with_rag(
    user_id: str,
    question: str,
    session_id: Optional[str] = None,
    use_history: bool = True,
    max_history: int = 5,
    top_k: int = 3
) -> dict:
    """
    Realiza una consulta de chat con RAG usando DeepSeek.

    Args:
        user_id: ID del usuario
        question: Pregunta del usuario
        session_id: ID de sesión (se genera uno nuevo si no se proporciona)
        use_history: Si usar el historial de conversación
        max_history: Número máximo de mensajes de historial
        top_k: Número de chunks relevantes a recuperar

    Returns:
        dict: Respuesta con answer, sources, session_id, etc.
    """
    try:
        # 1. Generar session_id si no existe
        if not session_id:
            session_id = f"session-{uuid.uuid4()}"

        # 2. Obtener chunks relevantes
        logger.info(f"Buscando contexto relevante para: '{question[:50]}...'")
        chunks_with_scores = get_relevant_chunks_with_scores(
            query=question,
            user_id=user_id,
            k=top_k
        )

        # 3. Formatear contexto
        context_parts = []
        sources = []

        for idx, (chunk, score) in enumerate(chunks_with_scores, 1):
            context_parts.append(f"[Fragmento {idx}]\n{chunk.page_content}\n")

            sources.append({
                "content": chunk.page_content[:200],  # Primeros 200 chars
                "document_id": chunk.metadata.get('document_id'),
                "filename": chunk.metadata.get('filename'),
                "chunk_index": chunk.metadata.get('chunk_index'),
                "relevance_score": float(score)
            })

        context = "\n---\n".join(context_parts) if context_parts else "No se encontró información relevante en los documentos."

        # 4. Obtener historial si está habilitado
        chat_history = []
        if use_history:
            history_tuples = get_chat_history_from_db(
                user_id=user_id,
                session_id=session_id,
                limit=max_history
            )
            chat_history = format_chat_history_for_prompt(history_tuples)

        # 5. Crear el prompt especializado en educación
        system_prompt = """Eres un **Asistente Pedagógico Inteligente** especializado en ayudar a docentes con la planificación de clases, diseño de ejercicios y selección de materiales educativos.

TU ROL:
- Ayudar a docentes a encontrar información relevante en guías pedagógicas, materiales educativos y documentos de planificación
- Proporcionar consejos prácticos basados en las mejores prácticas pedagógicas encontradas en los documentos
- Sugerir ejercicios, actividades y materiales didácticos
- Responder preguntas sobre planificación de clases, metodologías de enseñanza y recursos educativos

INSTRUCCIONES IMPORTANTES:
- Responde SOLO usando la información del CONTEXTO proporcionado
- SIEMPRE menciona la página específica de donde proviene la información (busca [Página X] en el contexto)
- Si el contexto menciona ejercicios, descríbelos detalladamente
- Si hay materiales o recursos, lista todos los mencionados
- Si el contexto no contiene información relevante, sugiere buscar en otras secciones o documentos
- Usa un tono profesional pero cercano, como un colega docente experimentado
- Estructura tus respuestas de forma clara con bullets o números cuando sea apropiado

FORMATO DE REFERENCIAS:
Cuando menciones información, indica la página así: "(ver página X)" o "según la página X"

CONTEXTO DE LOS DOCUMENTOS:
{context}

Ahora responde la pregunta del docente basándote en el contexto anterior, incluyendo siempre las referencias de página."""

        # 6. Llamar a DeepSeek API
        client = get_deepseek_client()

        # Construir mensajes
        messages = [
            {"role": "system", "content": system_prompt.format(context=context)}
        ]

        # Agregar historial
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        # Agregar pregunta actual
        messages.append({"role": "user", "content": question})

        logger.info(f"Llamando a DeepSeek API con {len(messages)} mensajes")

        # Hacer la llamada (sin streaming)
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=False
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else None

        logger.info(f"Respuesta generada: {len(answer)} caracteres, {tokens_used} tokens")

        # 7. Guardar en base de datos
        save_chat_to_db(
            user_id=user_id,
            message=question,
            response=answer,
            session_id=session_id,
            sources=sources
        )

        # 8. Retornar resultado
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "tokens_used": tokens_used,
            "model": DEEPSEEK_MODEL,
            "chunks_retrieved": len(chunks_with_scores)
        }

    except Exception as e:
        logger.error(f"Error en chat_with_rag: {e}")
        raise


def chat_with_rag_stream(
    user_id: str,
    question: str,
    session_id: Optional[str] = None,
    use_history: bool = True,
    max_history: int = 5,
    top_k: int = 3
):
    """
    Realiza chat con RAG usando streaming (Server-Sent Events).

    Args:
        user_id: ID del usuario
        question: Pregunta del usuario
        session_id: ID de sesión (opcional)
        use_history: Si usar historial de conversación
        max_history: Máximo de mensajes históricos
        top_k: Número de chunks relevantes a recuperar

    Yields:
        str: Chunks de la respuesta en formato SSE
    """
    import json

    try:
        # 1. Crear o usar sesión existente
        if not session_id:
            session_id = create_or_get_session(user_id)

        logger.info(f"Chat streaming - user: {user_id}, session: {session_id}, question: {question[:50]}...")

        # 2. Recuperar chunks relevantes
        chunks_with_scores = get_relevant_chunks_with_scores(
            query=question,
            user_id=user_id,
            k=top_k
        )

        if not chunks_with_scores:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No se encontraron documentos relevantes'})}\n\n"
            return

        # 3. Construir contexto
        context_parts = []
        sources = []

        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            # chunk es un objeto Document de LangChain
            context_parts.append(f"[Fragmento {i}]\n{chunk.page_content}")
            sources.append({
                "chunk_id": chunk.metadata.get("id", ""),
                "document_id": chunk.metadata.get("document_id", ""),
                "filename": chunk.metadata.get("filename", "Unknown"),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "relevance_score": round(score, 4)
            })

        context = "\n\n".join(context_parts)

        # Enviar fuentes primero
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

        # 4. Obtener historial si se requiere
        chat_history = []
        if use_history:
            history_tuples = get_chat_history_from_db(
                user_id=user_id,
                session_id=session_id,
                limit=max_history
            )
            chat_history = format_chat_history_for_prompt(history_tuples)

        # 5. Crear el prompt especializado
        system_prompt = """Eres un **Asistente Pedagógico Inteligente** especializado en ayudar a docentes con la planificación de clases, diseño de ejercicios y selección de materiales educativos.

TU ROL:
- Ayudar a docentes a encontrar información relevante en guías pedagógicas, materiales educativos y documentos de planificación
- Proporcionar consejos prácticos basados en las mejores prácticas pedagógicas encontradas en los documentos
- Sugerir ejercicios, actividades y materiales didácticos
- Responder preguntas sobre planificación de clases, metodologías de enseñanza y recursos educativos

INSTRUCCIONES IMPORTANTES:
- Responde SOLO usando la información del CONTEXTO proporcionado
- SIEMPRE menciona la página específica de donde proviene la información (busca [Página X] en el contexto)
- Si el contexto menciona ejercicios, descríbelos detalladamente
- Si hay materiales o recursos, lista todos los mencionados
- Si el contexto no contiene información relevante, sugiere buscar en otras secciones o documentos
- Usa un tono profesional pero cercano, como un colega docente experimentado
- Estructura tus respuestas de forma clara con bullets o números cuando sea apropiado

FORMATO DE REFERENCIAS:
Cuando menciones información, indica la página así: "(ver página X)" o "según la página X"

CONTEXTO DE LOS DOCUMENTOS:
{context}

Ahora responde la pregunta del docente basándote en el contexto anterior, incluyendo siempre las referencias de página."""

        # 6. Construir mensajes
        client = get_deepseek_client()

        messages = [
            {"role": "system", "content": system_prompt.format(context=context)}
        ]

        # Agregar historial
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        # Agregar pregunta actual
        messages.append({"role": "user", "content": question})

        logger.info(f"Llamando a DeepSeek API con streaming")

        # 7. Hacer la llamada con streaming
        full_response = ""

        stream = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )

        # 8. Enviar chunks conforme llegan
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content

                # Enviar chunk al cliente
                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

        logger.info(f"Streaming completado: {len(full_response)} caracteres")

        # 9. Guardar en base de datos
        save_chat_to_db(
            user_id=user_id,
            message=question,
            response=full_response,
            session_id=session_id,
            sources=sources
        )

        # 10. Enviar evento de finalización
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

    except Exception as e:
        logger.error(f"Error en chat_with_rag_stream: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"


def create_or_get_session(user_id: str, title: Optional[str] = None) -> str:
    """
    Crea una nueva sesión de chat o retorna una existente.

    Args:
        user_id: ID del usuario
        title: Título de la sesión (opcional)

    Returns:
        str: session_id
    """
    try:
        session_id = f"session-{uuid.uuid4()}"

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, title)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (session_id, user_id, title or "Nueva conversación"))
                conn.commit()

        logger.info(f"Nueva sesión creada: {session_id}")
        return session_id

    except Exception as e:
        logger.error(f"Error al crear sesión: {e}")
        # Retornar un session_id temporal
        return f"session-{uuid.uuid4()}"


def get_user_sessions(user_id: str, limit: int = 10) -> List[dict]:
    """
    Obtiene las sesiones de chat de un usuario.

    Args:
        user_id: ID del usuario
        limit: Número máximo de sesiones

    Returns:
        List[dict]: Lista de sesiones
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        cs.id,
                        cs.session_id,
                        cs.title,
                        cs.created_at,
                        cs.last_activity,
                        COUNT(ch.id) as message_count
                    FROM chat_sessions cs
                    LEFT JOIN chat_history ch ON ch.session_id = cs.session_id
                    WHERE cs.user_id = %s
                    GROUP BY cs.id
                    ORDER BY cs.last_activity DESC
                    LIMIT %s
                """, (user_id, limit))

                return cur.fetchall()

    except Exception as e:
        logger.error(f"Error al obtener sesiones: {e}")
        return []


if __name__ == "__main__":
    # Test del servicio
    import sys

    if len(sys.argv) > 2:
        test_user_id = sys.argv[1]
        test_question = " ".join(sys.argv[2:])

        print(f"💬 Chat con RAG")
        print(f"Usuario: {test_user_id}")
        print(f"Pregunta: {test_question}\n")

        result = chat_with_rag(
            user_id=test_user_id,
            question=test_question,
            top_k=3
        )

        print(f"🤖 Respuesta:\n{result['answer']}\n")
        print(f"📚 Fuentes ({len(result['sources'])}):")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['filename']} (chunk {source['chunk_index']}) - Score: {source['relevance_score']:.4f}")

        print(f"\n📊 Stats:")
        print(f"  - Session ID: {result['session_id']}")
        print(f"  - Tokens: {result['tokens_used']}")
        print(f"  - Model: {result['model']}")

    else:
        print("Uso: python chat_service.py <user_id> <pregunta>")
