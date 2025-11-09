"""
Text extraction module.
Extrae texto de archivos PDF usando pypdf.
"""

import os
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from loguru import logger


def extract_text_from_pdf(
    file_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None
) -> str:
    """
    Extrae texto de un archivo PDF.

    Args:
        file_path: Ruta al archivo PDF
        start_page: Página inicial (0-indexed). None = desde el inicio
        end_page: Página final (0-indexed). None = hasta el final

    Returns:
        str: Texto extraído del PDF

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no es un PDF válido
        Exception: Otros errores de extracción
    """
    # Validar que el archivo existe
    if not os.path.exists(file_path):
        error_msg = f"Archivo no encontrado: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Validar extensión
    file_extension = Path(file_path).suffix.lower()
    if file_extension != '.pdf':
        error_msg = f"El archivo no es un PDF: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Leer el PDF
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        logger.info(f"Procesando PDF: {file_path} ({total_pages} páginas)")

        # Determinar rango de páginas
        start = start_page if start_page is not None else 0
        end = end_page if end_page is not None else total_pages

        # Validar rango
        if start < 0 or end > total_pages or start >= end:
            error_msg = f"Rango de páginas inválido: {start}-{end} (total: {total_pages})"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Extraer texto página por página
        extracted_text = []

        for page_num in range(start, end):
            try:
                page = reader.pages[page_num]
                text = page.extract_text()

                if text and text.strip():
                    # Agregar marcador de página en el texto
                    page_text = f"[Página {page_num + 1}]\n{text}"
                    extracted_text.append(page_text)
                    logger.debug(f"Página {page_num + 1}: {len(text)} caracteres extraídos")
                else:
                    logger.warning(f"Página {page_num + 1}: Sin texto extraíble")

            except Exception as e:
                logger.warning(f"Error al extraer texto de página {page_num + 1}: {e}")
                continue

        # Unir todo el texto con marcadores de página
        full_text = "\n\n".join(extracted_text)

        if not full_text.strip():
            logger.warning(f"No se extrajo texto del PDF: {file_path}")
            return ""

        logger.info(f"Extracción completada: {len(full_text)} caracteres totales")
        return full_text

    except Exception as e:
        error_msg = f"Error al procesar PDF {file_path}: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def get_pdf_metadata(file_path: str) -> dict:
    """
    Obtiene metadata de un archivo PDF.

    Args:
        file_path: Ruta al archivo PDF

    Returns:
        dict: Diccionario con metadata del PDF
    """
    try:
        reader = PdfReader(file_path)

        metadata = {
            "num_pages": len(reader.pages),
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
        }

        # Agregar metadata del PDF si existe
        if reader.metadata:
            pdf_info = reader.metadata
            metadata.update({
                "title": pdf_info.get("/Title", ""),
                "author": pdf_info.get("/Author", ""),
                "subject": pdf_info.get("/Subject", ""),
                "creator": pdf_info.get("/Creator", ""),
                "producer": pdf_info.get("/Producer", ""),
            })

        return metadata

    except Exception as e:
        logger.error(f"Error al obtener metadata de {file_path}: {e}")
        return {
            "num_pages": 0,
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "error": str(e)
        }


def validate_pdf_file(file_path: str, max_size_mb: int = 50) -> tuple[bool, str]:
    """
    Valida un archivo PDF.

    Args:
        file_path: Ruta al archivo PDF
        max_size_mb: Tamaño máximo permitido en MB (default: 50)

    Returns:
        tuple: (es_válido, mensaje_error)
    """
    # Verificar que existe
    if not os.path.exists(file_path):
        return False, f"Archivo no encontrado: {file_path}"

    # Verificar extensión
    if Path(file_path).suffix.lower() != '.pdf':
        return False, "El archivo no es un PDF"

    # Verificar tamaño
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"Archivo muy grande: {file_size_mb:.2f}MB (máximo: {max_size_mb}MB)"

    # Intentar leer el PDF
    try:
        reader = PdfReader(file_path)
        if len(reader.pages) == 0:
            return False, "El PDF no contiene páginas"
        return True, "Válido"

    except Exception as e:
        return False, f"PDF corrupto o inválido: {str(e)}"


if __name__ == "__main__":
    # Test del módulo
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        is_valid, message = validate_pdf_file(pdf_path)

        if is_valid:
            print(f"PDF válido: {pdf_path}")
            metadata = get_pdf_metadata(pdf_path)
            print(f"📄 Metadata: {metadata}")

            text = extract_text_from_pdf(pdf_path)
            print(f"📝 Texto extraído: {len(text)} caracteres")
            print(f"Primeros 500 caracteres:\n{text[:500]}...")
        else:
            print(f"PDF inválido: {message}")
    else:
        print("Uso: python text_extractor.py <ruta_al_pdf>")
