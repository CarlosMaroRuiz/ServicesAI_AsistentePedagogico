# Arquitectura General del Sistema RAG + ML

El siguiente diagrama representa los tres flujos principales del sistema, que integran los servicios de **procesamiento de documentos**, **recuperación aumentada (RAG)** y **análisis mediante Machine Learning**.

![Architecture Diagram](resources/architecture_diagram.png)

---

## Flujo 1: Upload de Documentos (Azul)

**Objetivo:** Procesar y almacenar documentos PDF para su posterior consulta e integración en el sistema de recuperación de información.

**Pasos del flujo:**

1. El usuario sube un archivo PDF al endpoint **`/upload`**.
2. El **Ingest Service** procesa el archivo:

   * Guarda el archivo físico en el directorio `temp/`.
   * Extrae el texto del documento.
   * Divide el texto en *chunks*.
   * Genera **embeddings (384D)**.
3. Los embeddings y metadatos se almacenan en **PostgreSQL**.
4. Se actualizan las siguientes tablas:

   * `documents`
   * `langchain_pg_embedding`
   * `langchain_pg_collection`

---

## Flujo 2: Chat con RAG (Verde)

**Objetivo:** Permitir al usuario interactuar con el sistema mediante consultas naturales sobre los documentos procesados.

**Pasos del flujo:**

1. El usuario envía una consulta al endpoint **`/chat`**.
2. El **Chat Service** coordina el proceso de recuperación y generación.
3. El **Query Service** realiza una **búsqueda vectorial** en `langchain_pg_embedding`.
4. Se recuperan los **Top-K documentos** más relevantes.
5. Se construye un **prompt con contexto** y se envía a la **DeepSeek API (LLM externo)**.
6. La respuesta generada se devuelve al usuario a través del servicio de chat.

---

## Flujo 3: Análisis ML (Rojo)

**Objetivo:** Ejecutar análisis avanzados sobre los embeddings almacenados utilizando técnicas de Machine Learning.

**Pasos del flujo:**

1. El usuario solicita un análisis mediante los endpoints **`/ml/*`**.
2. La **LLM API** envía una petición TCP al puerto **`5555`**.
3. El **TCP Server (services_ML)** recibe la solicitud y procesa los datos.
4. Se ejecutan los siguientes procesos de ML:

   * **Clustering:** `HDBSCAN`
   * **Topic Modeling:** `BERTopic`
   * **Recommendations:** Basadas en similitud
   * **Visualization:** `UMAP 2D`
5. Se leen los embeddings desde PostgreSQL.
6. Los resultados se almacenan en las siguientes tablas:

   * `ml_clusters`
   * `ml_topics`
   * `ml_visualizations`
   * `ml_recommendations`
7. El servidor **responde vía TCP** al cliente con los resultados procesados.
8. Los resultados se devuelven finalmente al usuario.

---

## Componentes Clave

| Componente               | Descripción                                                   | Puerto         |
| ------------------------ | ------------------------------------------------------------- | -------------- |
| **services_LLM**         | Servicio principal de orquestación y endpoints REST (FastAPI) | `8000`         |
| **services_ML**          | Servicio para análisis ML (FastAPI + TCP Server)              | `8001`, `5555` |
| **PostgreSQL (AWS RDS)** | Base de datos con soporte para **pgvector**                   | —              |
| **DeepSeek API**         | Modelo LLM externo para generación de texto contextualizado   | —              |
| **temp/**                | Carpeta temporal para almacenamiento físico de PDFs           | —              |

