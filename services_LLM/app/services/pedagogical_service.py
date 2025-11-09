"""
Servicio de transformación pedagógica.
Extrae consejos pedagógicos, ejercicios y materiales de documentos usando DeepSeek API.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from psycopg.types.json import Json

from app.db.connection import get_db_connection

# Cargar variables de entorno
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_deepseek_client() -> OpenAI:
    """Obtiene el cliente de DeepSeek API."""
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )


def extract_pedagogical_content(text: str, filename: str) -> Dict:
    """
    Extrae consejos pedagógicos, ejercicios y materiales de un texto usando DeepSeek.

    Args:
        text: Texto del documento
        filename: Nombre del archivo

    Returns:
        dict: Contenido pedagógico estructurado
    """
    try:
        client = get_deepseek_client()

        prompt = f"""Analiza el siguiente documento educativo y extrae la siguiente información de manera estructurada:

1. **Consejos Pedagógicos**: Identifica y lista todos los consejos, recomendaciones o mejores prácticas para docentes.
2. **Ejercicios**: Identifica y lista todos los ejercicios, actividades o tareas propuestas.
3. **Materiales**: Identifica y lista todos los recursos, materiales didácticos o herramientas mencionadas.
4. **Objetivos de Aprendizaje**: Identifica los objetivos, competencias o resultados esperados.
5. **Estrategias de Enseñanza**: Metodologías, enfoques o técnicas de enseñanza propuestas.

Para cada item, incluye:
- Una descripción clara y concisa
- La página donde se encuentra (busca los marcadores [Página X])
- Nivel de relevancia (alta, media, baja)

Formato de respuesta:
```json
{{
  "consejos_pedagogicos": [
    {{"descripcion": "...", "pagina": X, "relevancia": "alta"}},
  ],
  "ejercicios": [
    {{"descripcion": "...", "tipo": "...", "pagina": X, "relevancia": "alta"}},
  ],
  "materiales": [
    {{"descripcion": "...", "tipo": "...", "pagina": X}},
  ],
  "objetivos_aprendizaje": [
    {{"descripcion": "...", "pagina": X}},
  ],
  "estrategias_ensenanza": [
    {{"descripcion": "...", "pagina": X}},
  ],
  "resumen_general": "Breve resumen del contenido pedagógico del documento"
}}
```

DOCUMENTO:
{text[:15000]}

Responde SOLO con el JSON, sin texto adicional."""

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "Eres un experto en pedagogía y análisis de material educativo. Tu tarea es extraer contenido pedagógico estructurado de documentos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        content = response.choices[0].message.content

        # Intentar parsear como JSON
        import json
        try:
            # Limpiar el contenido si viene con markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            pedagogical_data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("No se pudo parsear JSON, usando respuesta raw")
            pedagogical_data = {
                "raw_response": content,
                "resumen_general": content[:500]
            }

        logger.info(f"Contenido pedagógico extraído de {filename}")
        return pedagogical_data

    except Exception as e:
        logger.error(f"Error al extraer contenido pedagógico: {e}")
        return {
            "error": str(e),
            "resumen_general": "No se pudo procesar el contenido pedagógico"
        }


def save_pedagogical_content(document_id: str, pedagogical_data: Dict) -> None:
    """
    Guarda el contenido pedagógico en la base de datos.

    Args:
        document_id: UUID del documento
        pedagogical_data: Datos pedagógicos extraídos
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Actualizar metadata del documento con contenido pedagógico
                cur.execute("""
                    UPDATE documents
                    SET metadata = metadata || %s::jsonb
                    WHERE id = %s
                """, (Json({"pedagogical_content": pedagogical_data}), document_id))
                conn.commit()

        logger.info(f"Contenido pedagógico guardado para documento {document_id}")

    except Exception as e:
        logger.error(f"Error al guardar contenido pedagógico: {e}")
        raise


def get_pedagogical_content(document_id: str) -> Dict:
    """
    Obtiene el contenido pedagógico de un documento.

    Args:
        document_id: UUID del documento

    Returns:
        dict: Contenido pedagógico o None
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT metadata->'pedagogical_content' as content
                    FROM documents
                    WHERE id = %s
                """, (document_id,))

                result = cur.fetchone()
                if result and result['content']:
                    return result['content']
                return None

    except Exception as e:
        logger.error(f"Error al obtener contenido pedagógico: {e}")
        return None


def search_pedagogical_content(
    user_id: str,
    query_type: str,  # "consejos", "ejercicios", "materiales", "objetivos", "estrategias"
    keyword: str = None
) -> List[Dict]:
    """
    Busca contenido pedagógico específico en los documentos del usuario.

    Args:
        user_id: ID del usuario
        query_type: Tipo de contenido a buscar
        keyword: Palabra clave opcional para filtrar

    Returns:
        list: Lista de contenido pedagógico encontrado
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Mapear tipo a campo en JSON
                field_map = {
                    "consejos": "consejos_pedagogicos",
                    "ejercicios": "ejercicios",
                    "materiales": "materiales",
                    "objetivos": "objetivos_aprendizaje",
                    "estrategias": "estrategias_ensenanza"
                }

                field = field_map.get(query_type)
                if not field:
                    return []

                cur.execute(f"""
                    SELECT
                        d.id,
                        d.filename,
                        d.metadata->'pedagogical_content'->'{field}' as content
                    FROM documents d
                    WHERE d.user_id = %s
                    AND d.metadata->'pedagogical_content' IS NOT NULL
                    AND d.status = 'active'
                """, (user_id,))

                results = []
                for row in cur.fetchall():
                    if row['content']:
                        results.append({
                            "document_id": str(row['id']),
                            "filename": row['filename'],
                            "content": row['content']
                        })

                return results

    except Exception as e:
        logger.error(f"Error al buscar contenido pedagógico: {e}")
        return []


if __name__ == "__main__":
    # Test
    test_text = """
    [Página 1]
    Guía para Docentes

    Consejos pedagógicos:
    - Use ejemplos prácticos para explicar conceptos abstractos
    - Fomente la participación activa de los estudiantes

    Ejercicio 1: Resolución de problemas matemáticos
    Materiales necesarios: calculadora, papel
    """

    result = extract_pedagogical_content(test_text, "test.pdf")
    print(result)
