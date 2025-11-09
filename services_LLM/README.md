# Sistema RAG Conversacional con DeepSeek API

API REST completa de chat conversacional con **RAG (Retrieval-Augmented Generation)** para asistencia pedagógica inteligente, utilizando **DeepSeek API**, **LangChain**, **FastAPI** y **PostgreSQL con pgvector**.

---

## ¿Qué es esta API?

Esta API implementa un **sistema RAG (Retrieval-Augmented Generation)** especializado en contenido pedagógico. Permite a los docentes:

1. **Subir documentos PDF** (hasta 50 MB) con materiales educativos
2. **Procesar y vectorizar** el contenido automáticamente usando embeddings
3. **Consultar** el contenido mediante chat conversacional en lenguaje natural
4. **Recibir respuestas contextualizadas** generadas por IA, basadas en los documentos subidos
5. **Obtener referencias precisas** de las páginas de donde proviene la información
6. **Extraer contenido pedagógico** estructurado (ejercicios, consejos, materiales, estrategias)
7. **Mantener conversaciones** con historial contextualizado por sesiones
8. **Streaming en tiempo real** para ver las respuestas generarse palabra por palabra

La API combina la capacidad de búsqueda semántica en documentos (retrieval) con la generación de texto inteligente (generation), permitiendo que el bot responda preguntas basándose exclusivamente en el contenido de los documentos proporcionados.

---

## Características Principales

### Gestión de Documentos
- **Carga dinámica de PDFs**: Sube hasta 50 MB por documento
- **Actualización y eliminación**: Soft delete y hard delete
- **Procesamiento automático**: Extracción de texto, división en chunks, generación de embeddings
- **Metadata rica**: Información de páginas, tamaños, fechas, cantidad de chunks

### Búsqueda y Recuperación (RAG)
- **Embeddings locales**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones)
- **Vector Store**: PostgreSQL con extensión `pgvector` para búsqueda por similitud
- **Similarity Search**: Recuperación de fragmentos relevantes usando distancia coseno
- **Ranking por relevancia**: Scores de similitud para cada fragmento recuperado

### Chat Conversacional
- **LLM potente**: DeepSeek API (`deepseek-chat`) para generación de respuestas
- **Historial contextualizado**: Mantiene conversaciones coherentes
- **Sesiones de chat**: Agrupa conversaciones por temas o contextos
- **Streaming SSE**: Respuestas en tiempo real palabra por palabra (Server-Sent Events)
- **Referencias de página**: Cada respuesta incluye las fuentes con número de página

### Contenido Pedagógico Especializado
- **Extracción automática** de: consejos pedagógicos, ejercicios, materiales didácticos, objetivos, estrategias
- **Búsqueda por tipo**: Filtra contenido por categoría pedagógica
- **Marcadores de página**: Formato `[Página X]` en texto extraído

---

## Arquitectura del Sistema RAG

### Flujo General

```
┌─────────────┐
│   Usuario   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│              FastAPI (API REST)                 │
│  ┌──────────────┐        ┌──────────────┐      │
│  │Document Routes│        │ Chat Routes  │      │
│  └──────┬───────┘        └──────┬───────┘      │
│         │                       │               │
│         ▼                       ▼               │
│  ┌──────────────────────────────────────┐      │
│  │          LangChain Core              │      │
│  │  • Document Loaders                  │      │
│  │  • Text Splitters (Chunking)         │      │
│  │  • Embeddings (HuggingFace)          │      │
│  │  • Vector Store (PGVector)           │      │
│  │  • Prompt Templates                  │      │
│  └──────────────┬───────────────────────┘      │
└─────────────────┼───────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
┌─────────────┐      ┌─────────────────┐
│ PostgreSQL  │      │  DeepSeek API   │
│ + pgvector  │      │  (LLM Cloud)    │
│             │      │                 │
│ • Embeddings│      │ • Text Gen      │
│ • Metadata  │      │ • Streaming     │
│ • History   │      │ • Chat Model    │
└─────────────┘      └─────────────────┘
```

### Flujo de Carga de Documentos

```
1. Usuario sube PDF
   ↓
2. PyPDF extrae texto por páginas
   ↓
3. RecursiveCharacterTextSplitter divide en chunks
   (Tamaño: 1000 caracteres, Overlap: 200)
   ↓
4. HuggingFaceEmbeddings genera vectores (384 dims)
   ↓
5. PGVector almacena en PostgreSQL
   (Tabla: langchain_pg_embedding)
```

### Flujo de Chat Conversacional (RAG)

```
1. Usuario envía pregunta
   ↓
2. Sistema genera embedding de la pregunta
   ↓
3. Búsqueda por similitud en pgvector
   (Recupera top-k chunks más relevantes)
   ↓
4. Recupera historial de conversación (sesión)
   ↓
5. Construye prompt estructurado:
   [CONTEXTO de chunks] + [HISTORIAL] + [PREGUNTA]
   ↓
6. Envía a DeepSeek API
   ↓
7. Recibe respuesta (normal o streaming)
   ↓
8. Guarda intercambio en historial
   ↓
9. Retorna respuesta + fuentes al usuario
```

### Flujo de Streaming en Tiempo Real

```
1. Cliente hace POST a /chat/stream
   ↓
2. API abre conexión SSE (text/event-stream)
   ↓
3. Envía evento: {"type": "sources", "content": [...]}
   ↓
4. Llama a DeepSeek con stream=True
   ↓
5. Por cada chunk del LLM:
   Envía evento: {"type": "content", "content": "palabra"}
   ↓
6. Al finalizar:
   Guarda en historial
   Envía evento: {"type": "done", "session_id": "..."}
   ↓
7. Cliente cierra conexión
```

---

## Tecnologías, Modelos y Embedders

### Framework Principal
- **FastAPI**: Framework web moderno y rápido para Python
  - Validación automática con Pydantic
  - Documentación automática (Swagger/ReDoc)
  - Soporte nativo para async/await
  - CORS configurado para desarrollo

### LangChain
Framework orquestador para aplicaciones LLM:
- **Document Loaders**: Carga y procesa PDFs
- **Text Splitters**: Divide documentos en chunks procesables
- **Embeddings Integration**: Conecta modelos de embeddings
- **Vector Stores**: Abstracción para bases de datos vectoriales
- **Prompt Templates**: Construcción estructurada de prompts
- **Memory**: Gestión de historial conversacional

### Modelo de Lenguaje (LLM)
**DeepSeek API - deepseek-chat**
- **Proveedor**: DeepSeek AI (https://api.deepseek.com)
- **Modelo**: `deepseek-chat`
- **Capacidades**:
  - Generación de texto de alta calidad en español
  - Streaming con Server-Sent Events
  - Contexto largo (hasta ~8K tokens en prompts)
  - Temperatura configurable (0.7 por defecto)
  - Max tokens: 1000 por respuesta
- **Uso**: Generación de respuestas conversacionales basadas en contexto
- **API Compatible**: OpenAI SDK (usa `openai.OpenAI`)

### Modelo de Embeddings (Embedder)
**sentence-transformers/all-MiniLM-L6-v2**
- **Tipo**: Transformer pre-entrenado para embeddings semánticos
- **Dimensiones**: 384 (vectores de 384 números flotantes)
- **Ventajas**:
  - Ejecuta localmente (no requiere API externa)
  - Rápido y eficiente
  - Excelente para búsqueda semántica en español/inglés
  - Tamaño pequeño (~80 MB)
- **Uso**: Convierte texto en vectores numéricos para búsqueda por similitud
- **Biblioteca**: `sentence-transformers` de HuggingFace
- **Integración**: `HuggingFaceEmbeddings` de LangChain

### Base de Datos Vectorial
**PostgreSQL 14+ con extensión pgvector**
- **pgvector**: Extensión para almacenar y buscar vectores
- **Operaciones**:
  - Búsqueda por similitud usando distancia coseno
  - Índices vectoriales para búsquedas rápidas
  - Almacenamiento de embeddings de 384 dimensiones
- **Tablas principales**:
  - `langchain_pg_embedding`: Chunks + vectores + metadata
  - `langchain_pg_collection`: Colecciones por usuario
  - `chat_history`: Historial de conversaciones
  - `chat_sessions`: Sesiones de chat
  - `documents`: Metadata de documentos
  - `pedagogical_content`: Contenido educativo extraído

### Procesamiento de PDFs
- **pypdf**: Extracción de texto de documentos PDF
- **RecursiveCharacterTextSplitter**: División inteligente de texto
  - Chunk size: 1000 caracteres
  - Chunk overlap: 200 caracteres
  - Separadores: párrafos, líneas, palabras

### Utilidades
- **Loguru**: Logging mejorado con rotación y colores
- **psycopg**: Driver moderno de PostgreSQL
- **python-dotenv**: Gestión de variables de entorno
- **Pydantic**: Validación de datos y modelos

## Requisitos Previos

1. **Python 3.10+**
2. **PostgreSQL 14+** con extensión `pgvector`
3. **API Key de DeepSeek** ([obtener aquí](https://platform.deepseek.com/))

### Instalar PostgreSQL con pgvector

#### Windows
```bash
# Descargar e instalar PostgreSQL desde:
# https://www.postgresql.org/download/windows/

# Luego instalar pgvector desde:
# https://github.com/pgvector/pgvector
```

#### Linux/Ubuntu
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt install postgresql-14-pgvector
```

#### macOS
```bash
brew install postgresql@14
brew install pgvector
```

## Instalación

### 1. Clonar o crear el proyecto

```bash
cd llm_api
```

### 2. Crear entorno virtual

```bash
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar PostgreSQL

```bash
# Conectar a PostgreSQL
psql -U postgres

# Crear base de datos
CREATE DATABASE ragdb;

# Conectar a la base de datos
\c ragdb

# Habilitar extensión pgvector
CREATE EXTENSION vector;

# Salir
\q
```

### 5. Configurar variables de entorno

Edita el archivo `.env` con tus credenciales:

```bash
# API Keys
DEEPSEEK_API_KEY=tu-api-key-aqui
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ragdb
DB_USER=postgres
DB_PASSWORD=tu-password
DATABASE_URL=postgresql+psycopg://postgres:tu-password@localhost:5432/ragdb

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# DeepSeek Model
DEEPSEEK_MODEL=deepseek-chat

# Application Settings
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=3
```

### 6. Inicializar la base de datos

```bash
# El schema se crea automáticamente al iniciar la app, o manualmente:
psql -U postgres -d ragdb -f app/db/schema.sql
```

## Ejecución

### Iniciar el servidor

```bash
# Modo desarrollo (con auto-reload)
python app/main.py

# O usando uvicorn directamente
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en: **http://localhost:8000**

- **Documentación API (Swagger)**: http://localhost:8000/docs
- **Documentación alternativa (ReDoc)**: http://localhost:8000/redoc

---

## API Endpoints Completa

La API expone **19 endpoints** organizados en 3 categorías principales.

### Sistema y Salud

#### `GET /health`
Verifica que el servidor esté funcionando.

**Respuesta:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-06T10:30:00",
  "database": "connected"
}
```

**Ejemplo:**
```bash
curl http://localhost:8000/health
```

#### `GET /info`
Información general del sistema y configuración.

**Respuesta:**
```json
{
  "name": "RAG Pedagogical API",
  "version": "1.0.0",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "deepseek-chat",
  "max_file_size_mb": 50,
  "chunk_size": 1000
}
```

---

### Gestión de Documentos

#### `POST /documents/upload`
Sube un documento PDF y lo procesa automáticamente.

**Parámetros Query:**
- `user_id` (required): ID del usuario propietario

**Body:**
- `file` (multipart/form-data): Archivo PDF (máx 50 MB)

**Respuesta:**
```json
{
  "success": true,
  "message": "Documento subido y procesado exitosamente",
  "document_id": "uuid-123",
  "filename": "documento.pdf",
  "chunks_created": 25,
  "file_size_mb": 2.5,
  "pages": 10
}
```

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/documents/upload?user_id=docente01" \
  -F "file=@materiales.pdf"
```

---

#### `GET /documents/list`
Lista todos los documentos de un usuario.

**Parámetros Query:**
- `user_id` (required): ID del usuario
- `include_deleted` (optional): Incluir documentos eliminados (default: false)

**Respuesta:**
```json
{
  "total": 3,
  "documents": [
    {
      "id": "uuid-123",
      "user_id": "docente01",
      "filename": "materiales.pdf",
      "file_size_mb": 2.5,
      "chunks_count": 25,
      "pages": 10,
      "created_at": "2025-01-06T10:00:00",
      "is_deleted": false
    }
  ]
}
```

**Ejemplo:**
```bash
curl "http://localhost:8000/documents/list?user_id=docente01"
```

---

#### `GET /documents/{document_id}`
Obtiene información detallada de un documento específico.

**Parámetros Path:**
- `document_id` (required): UUID del documento

**Respuesta:**
```json
{
  "id": "uuid-123",
  "user_id": "docente01",
  "filename": "materiales.pdf",
  "file_size_mb": 2.5,
  "chunks_count": 25,
  "pages": 10,
  "created_at": "2025-01-06T10:00:00",
  "updated_at": "2025-01-06T10:05:00",
  "is_deleted": false,
  "metadata": {
    "original_filename": "materiales_didacticos.pdf"
  }
}
```

---

#### `PUT /documents/{document_id}`
Actualiza un documento existente con un nuevo archivo.

**Parámetros Path:**
- `document_id` (required): UUID del documento

**Body:**
- `file` (multipart/form-data): Nuevo archivo PDF

**Respuesta:**
```json
{
  "success": true,
  "message": "Documento actualizado exitosamente",
  "document_id": "uuid-123",
  "chunks_created": 30,
  "previous_chunks_deleted": 25
}
```

**Ejemplo:**
```bash
curl -X PUT "http://localhost:8000/documents/uuid-123" \
  -F "file=@materiales_v2.pdf"
```

---

#### `DELETE /documents/{document_id}`
Elimina un documento (soft delete por defecto).

**Parámetros Path:**
- `document_id` (required): UUID del documento

**Parámetros Query:**
- `hard_delete` (optional): Eliminación permanente (default: false)

**Respuesta (Soft Delete):**
```json
{
  "success": true,
  "message": "Documento marcado como eliminado",
  "document_id": "uuid-123",
  "chunks_deleted": 0
}
```

**Respuesta (Hard Delete):**
```json
{
  "success": true,
  "message": "Documento eliminado permanentemente",
  "document_id": "uuid-123",
  "chunks_deleted": 25
}
```

**Ejemplo:**
```bash
# Soft delete
curl -X DELETE "http://localhost:8000/documents/uuid-123"

# Hard delete
curl -X DELETE "http://localhost:8000/documents/uuid-123?hard_delete=true"
```

---

#### `PATCH /documents/{document_id}/rename`
Renombra un documento.

**Parámetros Path:**
- `document_id` (required): UUID del documento

**Body:**
```json
{
  "new_filename": "materiales_didacticos_2025.pdf"
}
```

**Respuesta:**
```json
{
  "success": true,
  "message": "Documento renombrado exitosamente",
  "old_filename": "materiales.pdf",
  "new_filename": "materiales_didacticos_2025.pdf"
}
```

---

#### `POST /documents/{document_id}/restore`
Restaura un documento eliminado (soft deleted).

**Parámetros Path:**
- `document_id` (required): UUID del documento

**Respuesta:**
```json
{
  "success": true,
  "message": "Documento restaurado exitosamente",
  "document_id": "uuid-123"
}
```

---

### Contenido Pedagógico

#### `POST /documents/{document_id}/extract-pedagogical`
Extrae contenido pedagógico estructurado de un documento.

**Parámetros Path:**
- `document_id` (required): UUID del documento

**Respuesta:**
```json
{
  "success": true,
  "document_id": "uuid-123",
  "extracted_items": 15,
  "categories": {
    "consejos": 5,
    "ejercicios": 7,
    "materiales": 3
  }
}
```

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/documents/uuid-123/extract-pedagogical"
```

---

#### `GET /documents/{document_id}/pedagogical`
Obtiene el contenido pedagógico extraído de un documento.

**Parámetros Path:**
- `document_id` (required): UUID del documento

**Parámetros Query:**
- `content_type` (optional): Filtrar por tipo (consejos, ejercicios, materiales, objetivos, estrategias)

**Respuesta:**
```json
{
  "document_id": "uuid-123",
  "total": 15,
  "items": [
    {
      "id": "ped-001",
      "content_type": "ejercicios",
      "title": "Ejercicio de comprensión lectora",
      "content": "Leer el texto y responder las siguientes preguntas...",
      "page_reference": "[Página 5]",
      "created_at": "2025-01-06T10:10:00"
    }
  ]
}
```

---

#### `GET /documents/pedagogical/search`
Busca contenido pedagógico en todos los documentos del usuario.

**Parámetros Query:**
- `user_id` (required): ID del usuario
- `query` (required): Texto de búsqueda
- `content_type` (optional): Filtrar por tipo
- `limit` (optional): Máximo de resultados (default: 10)

**Respuesta:**
```json
{
  "total": 5,
  "results": [
    {
      "content_type": "consejos",
      "title": "Estrategia de enseñanza activa",
      "content": "Para fomentar la participación...",
      "page_reference": "[Página 12]",
      "filename": "guia_docente.pdf",
      "relevance_score": 0.92
    }
  ]
}
```

**Ejemplo:**
```bash
curl "http://localhost:8000/documents/pedagogical/search?user_id=docente01&query=ejercicios%20de%20lectura&content_type=ejercicios"
```

---

### Chat Conversacional

#### `POST /chat/`
Realiza una consulta de chat con RAG (sin streaming).

**Body:**
```json
{
  "user_id": "docente01",
  "message": "¿Qué ejercicios recomiendas para mejorar la comprensión lectora?",
  "session_id": "session-abc",
  "use_history": true,
  "max_history": 5,
  "top_k": 3
}
```

**Respuesta:**
```json
{
  "answer": "Según los documentos proporcionados, se recomiendan los siguientes ejercicios para mejorar la comprensión lectora:\n\n1. Lectura guiada con preguntas específicas [Página 5]\n2. Resúmenes progresivos de párrafos [Página 8]\n3. Mapas conceptuales del contenido [Página 12]",
  "sources": [
    {
      "content": "Fragmento del documento...",
      "document_id": "uuid-123",
      "filename": "guia_docente.pdf",
      "chunk_index": 5,
      "relevance_score": 0.89
    }
  ],
  "session_id": "session-abc",
  "timestamp": "2025-01-06T10:30:00",
  "tokens_used": 350,
  "model": "deepseek-chat"
}
```

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "docente01",
    "message": "¿Qué ejercicios de matemáticas puedo usar?",
    "use_history": true,
    "top_k": 3
  }'
```

---

#### `POST /chat/stream`
Realiza una consulta de chat con RAG usando **Server-Sent Events (SSE)** para streaming en tiempo real.

**Body:** (mismo que `/chat/`)

**Respuesta (SSE Stream):**
```
data: {"type": "sources", "content": [{"filename": "guia.pdf", "chunk_index": 5, "relevance_score": 0.89}]}

data: {"type": "content", "content": "Según"}

data: {"type": "content", "content": " los"}

data: {"type": "content", "content": " documentos"}

...

data: {"type": "done", "session_id": "session-abc"}
```

**Eventos posibles:**
- `sources`: Fuentes recuperadas del vector store
- `content`: Chunks de texto de la respuesta
- `done`: Finalización exitosa
- `error`: Error durante el procesamiento

**Ejemplo JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
    },
    body: JSON.stringify({
        user_id: 'docente01',
        message: '¿Qué materiales necesito?',
        use_history: true,
        top_k: 3
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'content') {
                console.log(data.content); // Mostrar palabra por palabra
            }
        }
    }
}
```

**Ejemplo Python:**
```python
import requests
import json

response = requests.post(
    'http://localhost:8000/chat/stream',
    json={
        'user_id': 'docente01',
        'message': '¿Qué estrategias pedagógicas recomiendas?',
        'use_history': True,
        'top_k': 3
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = json.loads(line_str[6:])

            if data['type'] == 'content':
                print(data['content'], end='', flush=True)
```

---

#### `POST /chat/session`
Crea una nueva sesión de chat.

**Body:**
```json
{
  "user_id": "docente01",
  "title": "Planificación de clase de literatura"
}
```

**Respuesta:**
```json
{
  "id": 1,
  "session_id": "session-xyz",
  "user_id": "docente01",
  "title": "Planificación de clase de literatura",
  "created_at": "2025-01-06T10:00:00",
  "last_activity": "2025-01-06T10:00:00",
  "message_count": 0
}
```

---

#### `GET /chat/sessions/{user_id}`
Lista las sesiones de chat de un usuario.

**Parámetros Path:**
- `user_id` (required): ID del usuario

**Parámetros Query:**
- `limit` (optional): Máximo de sesiones (default: 10, max: 100)

**Respuesta:**
```json
{
  "total": 5,
  "sessions": [
    {
      "id": 1,
      "session_id": "session-xyz",
      "title": "Planificación de clase de literatura",
      "created_at": "2025-01-06T10:00:00",
      "last_activity": "2025-01-06T11:30:00",
      "message_count": 12
    }
  ]
}
```

**Ejemplo:**
```bash
curl "http://localhost:8000/chat/sessions/docente01?limit=20"
```

---

#### `GET /chat/history/{user_id}`
Obtiene el historial de chat de un usuario.

**Parámetros Path:**
- `user_id` (required): ID del usuario

**Parámetros Query:**
- `session_id` (optional): Filtrar por sesión específica
- `limit` (optional): Máximo de mensajes (default: 20, max: 100)

**Respuesta:**
```json
{
  "total": 15,
  "history": [
    {
      "id": 1,
      "user_id": "docente01",
      "session_id": "session-xyz",
      "message": "¿Qué ejercicios recomiendas?",
      "response": "Recomiendo los siguientes ejercicios...",
      "sources": [...],
      "created_at": "2025-01-06T10:15:00"
    }
  ],
  "session_id": "session-xyz"
}
```

**Ejemplo:**
```bash
curl "http://localhost:8000/chat/history/docente01?session_id=session-xyz&limit=50"
```

---

#### `DELETE /chat/history/{user_id}`
Elimina el historial de chat.

**Parámetros Path:**
- `user_id` (required): ID del usuario

**Parámetros Query:**
- `session_id` (optional): Eliminar solo de una sesión específica

**Respuesta:**
```json
{
  "success": true,
  "message": "Eliminados 15 mensajes",
  "deleted_count": 15
}
```

**Ejemplo:**
```bash
# Eliminar todo el historial del usuario
curl -X DELETE "http://localhost:8000/chat/history/docente01"

# Eliminar solo una sesión
curl -X DELETE "http://localhost:8000/chat/history/docente01?session_id=session-xyz"
```

---

#### `GET /chat/session/{session_id}/messages`
Obtiene todos los mensajes de una sesión específica.

**Parámetros Path:**
- `session_id` (required): ID de la sesión

**Parámetros Query:**
- `limit` (optional): Máximo de mensajes (default: 50, max: 200)

**Respuesta:**
```json
{
  "session_id": "session-xyz",
  "total": 12,
  "messages": [
    {
      "id": 1,
      "message": "¿Qué materiales necesito?",
      "response": "Según los documentos...",
      "sources": [...],
      "created_at": "2025-01-06T10:15:00"
    }
  ]
}
```

**Ejemplo:**
```bash
curl "http://localhost:8000/chat/session/session-xyz/messages?limit=100"
```

## Ejemplos de Uso con Python

### Subir PDF

```python
import requests

url = "http://localhost:8000/documents/upload"
params = {"user_id": "user123"}
files = {"file": open("documento.pdf", "rb")}

response = requests.post(url, params=params, files=files)
print(response.json())
```

### Chat

```python
import requests

url = "http://localhost:8000/chat/"
data = {
    "user_id": "user123",
    "message": "¿Cuáles son las cláusulas importantes?",
    "use_history": True,
    "top_k": 3
}

response = requests.post(url, json=data)
result = response.json()

print(f"Respuesta: {result['answer']}")
print(f"\nFuentes:")
for i, source in enumerate(result['sources'], 1):
    print(f"{i}. {source['filename']} (chunk {source['chunk_index']})")
```

---

## Servicios (Services)

Los servicios son la capa de lógica de negocio que procesa las operaciones principales del sistema.

### `ingest_service.py`
**Responsabilidad**: Procesamiento e ingesta de documentos PDF.

**Funciones principales:**
- `ingest_pdf(file, user_id)`: Procesa un PDF completo
  1. Guarda archivo temporalmente
  2. Extrae texto por páginas usando pypdf
  3. Divide en chunks con RecursiveCharacterTextSplitter
  4. Genera embeddings con HuggingFaceEmbeddings
  5. Almacena en PostgreSQL con pgvector
  6. Guarda metadata en tabla `documents`
  7. Retorna documento_id y estadísticas

**Tecnologías usadas:**
- `pypdf.PdfReader`: Lectura de PDFs
- `LangChain.RecursiveCharacterTextSplitter`: División de texto
- `LangChain.HuggingFaceEmbeddings`: Generación de vectores
- `LangChain.PGVector`: Almacenamiento vectorial

---

### `query_service.py`
**Responsabilidad**: Búsqueda y recuperación de información del vector store.

**Funciones principales:**
- `get_relevant_chunks_with_scores(query, user_id, k=3)`:
  - Genera embedding de la consulta
  - Busca por similitud coseno en pgvector
  - Retorna top-k chunks con scores de relevancia
  - Filtra por user_id para aislar datos

- `search_documents(query, user_id, filters)`:
  - Búsqueda con filtros adicionales (tipo, fecha, documento)
  - Combina búsqueda vectorial con filtros SQL

**Tecnologías usadas:**
- `LangChain.PGVector.similarity_search_with_score()`: Búsqueda vectorial
- Consultas SQL para filtrado adicional

---

### `chat_service.py`
**Responsabilidad**: Orquestación del chat conversacional con RAG.

**Funciones principales:**
- `chat_with_rag(user_id, question, session_id, use_history, max_history, top_k)`:
  1. Crea o recupera sesión
  2. Obtiene chunks relevantes del vector store
  3. Recupera historial de conversación
  4. Construye prompt con contexto + historial + pregunta
  5. Llama a DeepSeek API
  6. Guarda intercambio en historial
  7. Retorna respuesta + fuentes

- `chat_with_rag_stream(...)`:
  - Versión streaming de chat_with_rag
  - Genera eventos SSE en tiempo real
  - Yield de chunks conforme DeepSeek los genera

- `create_or_get_session(user_id, title)`:
  - Gestión de sesiones de chat
  - Actualiza `last_activity` en cada mensaje

- `get_chat_history_from_db(user_id, session_id, limit)`:
  - Recupera conversaciones previas
  - Formatea para construcción de prompts

- `save_chat_to_db(user_id, question, response, session_id, sources)`:
  - Persiste intercambios de chat
  - Almacena fuentes en formato JSON

**Tecnologías usadas:**
- `openai.OpenAI`: Cliente para DeepSeek API
- `StreamingResponse`: FastAPI para SSE
- Construcción de prompts estructurados

---

### `pedagogical_service.py`
**Responsabilidad**: Extracción y gestión de contenido pedagógico.

**Funciones principales:**
- `extract_pedagogical_content(document_id)`:
  - Analiza chunks del documento
  - Detecta patrones pedagógicos (ejercicios, consejos, materiales)
  - Extrae referencias de página
  - Almacena en tabla `pedagogical_content`

- `get_pedagogical_content(document_id, content_type)`:
  - Recupera contenido educativo extraído
  - Filtra por tipo (consejos, ejercicios, materiales, objetivos, estrategias)

- `search_pedagogical(user_id, query, content_type)`:
  - Búsqueda semántica en contenido pedagógico
  - Combina embeddings con filtros de tipo

**Patrones detectados:**
- **Consejos**: "se recomienda", "es importante", "sugerencia"
- **Ejercicios**: "actividad", "ejercicio", "práctica"
- **Materiales**: "material necesario", "recursos", "herramientas"
- **Objetivos**: "objetivo", "meta", "lograr que"
- **Estrategias**: "estrategia", "metodología", "enfoque"

---

### `update_service.py`
**Responsabilidad**: Actualización de documentos existentes.

**Funciones principales:**
- `update_document(document_id, new_file)`:
  1. Elimina chunks antiguos del vector store
  2. Procesa nuevo PDF
  3. Genera nuevos embeddings
  4. Actualiza metadata del documento
  5. Preserva historial de chats asociados

**Tecnologías usadas:**
- Reutiliza `ingest_service.py` para procesamiento
- Transacciones SQL para atomicidad

---

### `delete_service.py`
**Responsabilidad**: Eliminación de documentos (soft/hard delete).

**Funciones principales:**
- `soft_delete_document(document_id)`:
  - Marca `is_deleted = true` en BD
  - Mantiene chunks y metadata
  - Reversible con `restore_document()`

- `hard_delete_document(document_id)`:
  - Elimina permanentemente de `documents`
  - Elimina chunks del vector store
  - Elimina contenido pedagógico
  - Mantiene historial de chats (opcional)

- `restore_document(document_id)`:
  - Restaura documentos soft-deleted
  - Marca `is_deleted = false`

**Tecnologías usadas:**
- Eliminación en cascada con SQL
- `PGVector.delete()` para chunks

---

## Rutas (Routes)

Las rutas definen los endpoints HTTP y conectan las peticiones con los servicios.

### `document_routes.py`
**Prefijo**: `/documents`
**Tag**: `documents`

**Endpoints expuestos:**
- `POST /upload` → `ingest_service.ingest_pdf()`
- `GET /list` → Consulta SQL a tabla `documents`
- `GET /{document_id}` → Consulta SQL + metadatos
- `PUT /{document_id}` → `update_service.update_document()`
- `DELETE /{document_id}` → `delete_service.soft_delete()` o `hard_delete()`
- `PATCH /{document_id}/rename` → UPDATE SQL
- `POST /{document_id}/restore` → `delete_service.restore_document()`
- `POST /{document_id}/extract-pedagogical` → `pedagogical_service.extract_pedagogical_content()`
- `GET /{document_id}/pedagogical` → `pedagogical_service.get_pedagogical_content()`
- `GET /pedagogical/search` → `pedagogical_service.search_pedagogical()`

**Validaciones:**
- Verificación de permisos por `user_id`
- Tamaño máximo de archivo (50 MB)
- Tipo MIME (application/pdf)
- Existencia de documento antes de operaciones

---

### `chat_routes.py`
**Prefijo**: `/chat`
**Tag**: `chat`

**Endpoints expuestos:**
- `POST /` → `chat_service.chat_with_rag()`
- `POST /stream` → `chat_service.chat_with_rag_stream()`
- `POST /session` → `chat_service.create_or_get_session()`
- `GET /sessions/{user_id}` → `chat_service.get_user_sessions()`
- `GET /history/{user_id}` → `chat_service.get_chat_history_from_db()`
- `DELETE /history/{user_id}` → DELETE SQL en `chat_history`
- `GET /session/{session_id}/messages` → Consulta SQL filtrada por sesión

**Modelos Pydantic usados:**
- `ChatRequest`: Validación de entrada
- `ChatResponse`: Formato de salida
- `ChatSessionCreate`, `ChatSessionResponse`
- `ChatHistoryResponse`

**Headers especiales (streaming):**
- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `Connection: keep-alive`
- `X-Accel-Buffering: no`

---

## Glosario de Conceptos RAG y LangChain

### RAG (Retrieval-Augmented Generation)
Técnica que combina **recuperación de información** (retrieval) con **generación de texto** (generation). En lugar de que el LLM responda solo desde su conocimiento pre-entrenado, primero busca información relevante en una base de datos (vector store) y luego genera una respuesta basada en esos documentos recuperados.

**Ventajas**:
- Respuestas basadas en datos actualizados o específicos del dominio
- Reducción de alucinaciones del LLM
- Trazabilidad (se pueden citar las fuentes)
- No requiere re-entrenar el modelo

---

### Embeddings (Vectorización)
Representación numérica de texto en forma de vectores de números flotantes. Textos con significados similares tienen vectores cercanos en el espacio vectorial.

**Ejemplo**:
```
"perro" → [0.2, 0.8, 0.1, ..., 0.5]  (384 dimensiones)
"gato"  → [0.3, 0.7, 0.2, ..., 0.4]  (vectores cercanos)
"casa"  → [0.9, 0.1, 0.8, ..., 0.2]  (vector lejano)
```

**En este proyecto**:
- Modelo: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensiones: 384 (cada texto → 384 números)
- Generación: Local, sin llamadas a API externa

---

### Chunks (Fragmentos)
División de documentos largos en piezas más pequeñas para procesamiento eficiente.

**¿Por qué chunking?**
- Los LLMs tienen límites de tokens en el contexto
- Búsqueda más precisa (recuperar secciones relevantes, no documentos completos)
- Mejor rendimiento de embeddings

**En este proyecto**:
- Chunk size: 1000 caracteres (~250 palabras)
- Overlap: 200 caracteres (para mantener contexto entre chunks)
- Método: `RecursiveCharacterTextSplitter`
- Separadores: Párrafos → Oraciones → Palabras

**Ejemplo**:
```
Documento original (3000 caracteres):
[...............................................]

Chunks generados:
Chunk 1: [0-1000] caracteres
Chunk 2: [800-1800] caracteres (overlap de 200 con chunk 1)
Chunk 3: [1600-2600] caracteres
Chunk 4: [2400-3000] caracteres
```

---

### Vector Store (Almacén Vectorial)
Base de datos especializada en almacenar y buscar vectores (embeddings).

**En este proyecto**:
- Tecnología: **PostgreSQL + pgvector**
- Operación principal: **Búsqueda por similitud**
- Métrica: **Distancia coseno** (0 = idénticos, 1 = opuestos)

**Flujo**:
1. Usuario pregunta: "¿Qué ejercicios de lectura hay?"
2. Sistema genera embedding de la pregunta
3. pgvector busca chunks con embeddings más cercanos
4. Retorna top-k chunks más relevantes (k=3 por defecto)

---

### Similarity Search (Búsqueda por Similitud)
Recuperación de documentos/chunks basada en cercanía semántica, no coincidencia exacta de palabras.

**Ejemplo**:
```
Pregunta: "actividades para mejorar comprensión de textos"

Chunks recuperados (sin contener las palabras exactas):
1. "Ejercicios de lectura reflexiva..." (score: 0.89)
2. "Estrategias para entender mejor los escritos..." (score: 0.85)
3. "Prácticas de análisis textual..." (score: 0.82)
```

**Ventaja sobre búsqueda tradicional**:
- Búsqueda tradicional: Coincidencia de palabras clave
- Similarity search: Comprende el significado semántico

---

### LangChain
Framework de Python para construir aplicaciones con LLMs. Proporciona abstracciones y herramientas para orquestar sistemas complejos.

**Componentes usados en este proyecto**:

#### **Document (Objeto)**
Estructura que representa un fragmento de texto con metadata.
```python
Document(
    page_content="Texto del fragmento...",
    metadata={
        "id": "uuid-123",
        "document_id": "doc-456",
        "filename": "guia.pdf",
        "page": 5,
        "chunk_index": 12
    }
)
```

#### **Text Splitters**
Dividen texto en chunks.
- `RecursiveCharacterTextSplitter`: Divide recursivamente por separadores
- Parámetros: `chunk_size`, `chunk_overlap`, `separators`

#### **Embeddings**
Interfaz para modelos de embeddings.
- `HuggingFaceEmbeddings`: Modelos de sentence-transformers
- Método: `embed_query()`, `embed_documents()`

#### **Vector Stores**
Almacenes vectoriales con interfaz unificada.
- `PGVector`: Integración con PostgreSQL + pgvector
- Métodos: `similarity_search()`, `add_documents()`, `delete()`

#### **Prompt Templates**
Construcción estructurada de prompts para LLMs.
```python
template = """
Contexto: {context}
Historial: {history}
Pregunta: {question}

Responde basándote en el contexto.
"""
```

---

### LLM (Large Language Model)
Modelo de lenguaje de gran tamaño entrenado para generar y comprender texto.

**En este proyecto**:
- Modelo: **DeepSeek Chat**
- Proveedor: DeepSeek AI
- Uso: Generación de respuestas conversacionales
- Parámetros:
  - Temperature: 0.7 (creatividad moderada)
  - Max tokens: 1000 (longitud máxima de respuesta)
  - Stream: True/False (streaming o respuesta completa)

---

### Tokens
Unidades mínimas en las que el LLM procesa texto. Aproximadamente:
- 1 token ≈ 4 caracteres en inglés
- 1 token ≈ 0.75 palabras
- 100 tokens ≈ 75 palabras

**Límites de contexto**:
- DeepSeek: ~8000 tokens de entrada
- Esto incluye: contexto + historial + pregunta + instrucciones

---

### SSE (Server-Sent Events)
Tecnología de comunicación unidireccional del servidor al cliente para transmisión de datos en tiempo real.

**Formato**:
```
data: {"type": "content", "content": "palabra"}\n\n
```

**En este proyecto**:
- Usado en `/chat/stream`
- Permite mostrar respuestas palabra por palabra
- Content-Type: `text/event-stream`
- Alternativa a WebSockets (más simple, unidireccional)

---

### Prompt Engineering
Arte de diseñar prompts efectivos para obtener mejores respuestas de LLMs.

**Prompt usado en este proyecto**:
```
Eres un asistente pedagógico. Usa el siguiente contexto de documentos educativos.

CONTEXTO:
{chunks recuperados del vector store}

HISTORIAL DE CONVERSACIÓN:
{últimos N mensajes}

PREGUNTA DEL USUARIO:
{pregunta actual}

INSTRUCCIONES:
- Responde basándote SOLO en el contexto proporcionado
- Incluye referencias de página [Página X]
- Si no sabes, di que no tienes información
- Sé conciso y pedagógico
```

---

### Metadata (Metadatos)
Información adicional sobre los chunks y documentos.

**Metadatos almacenados**:
- `document_id`: UUID del documento padre
- `filename`: Nombre del archivo PDF
- `page`: Número de página original
- `chunk_index`: Posición del chunk en el documento
- `user_id`: Propietario del documento
- `created_at`: Timestamp de creación

**Uso**:
- Filtrado de búsquedas por usuario
- Referencias de página en respuestas
- Trazabilidad de fuentes

---

### Cosine Distance (Distancia Coseno)
Métrica para medir similitud entre vectores. Calcula el ángulo entre dos vectores en el espacio multidimensional.

**Rango**: 0 (idénticos) a 1 (completamente diferentes)

**Fórmula conceptual**:
```
similitud = 1 - (ángulo entre vectores / 180°)
```

**En pgvector**:
```sql
SELECT * FROM embeddings
ORDER BY embedding <=> query_vector
LIMIT 3;  -- Top 3 más similares
```

---

### Session (Sesión de Chat)
Agrupación lógica de conversaciones relacionadas. Permite:
- Mantener contexto entre múltiples preguntas
- Organizar conversaciones por temas
- Recuperar historiales específicos

**Estructura**:
- `session_id`: Identificador único (UUID)
- `user_id`: Propietario de la sesión
- `title`: Título descriptivo
- `created_at`, `last_activity`: Timestamps

---

## Estructura del Proyecto

```
llm_api/
├── app/
│   ├── main.py                      # Punto de entrada FastAPI
│   ├── routes/
│   │   ├── document_routes.py       # 11 endpoints de documentos
│   │   └── chat_routes.py           # 8 endpoints de chat
│   ├── services/
│   │   ├── ingest_service.py        # Procesamiento de PDFs
│   │   ├── query_service.py         # Búsqueda en vector store
│   │   ├── chat_service.py          # Chat RAG + streaming
│   │   ├── pedagogical_service.py   # Extracción pedagógica
│   │   ├── update_service.py        # Actualización de docs
│   │   └── delete_service.py        # Eliminación soft/hard
│   ├── db/
│   │   ├── connection.py            # Pool de conexiones PostgreSQL
│   │   └── schema.sql               # Esquema de BD (6 tablas)
│   ├── utils/
│   │   ├── text_extractor.py        # Extracción de texto de PDFs
│   │   └── chunker.py               # División inteligente de texto
│   └── models/
│       ├── document_model.py        # Pydantic models para docs
│       ├── chat_model.py            # Pydantic models para chat
│       └── response_model.py        # Pydantic models de respuestas
├── temp/                            # PDFs temporales durante procesamiento
├── logs/                            # Logs con rotación diaria
│   └── app_YYYY-MM-DD.log
├── test_streaming.html              # Demo interactivo de streaming
├── test_streaming.py                # Script de prueba de streaming
├── .env                             # Variables de entorno
├── requirements.txt                 # Dependencias Python
└── README.md                        # Esta documentación
```

## Flujo de Trabajo

### Carga de Documentos
1. Usuario sube PDF → `/documents/upload`
2. Sistema extrae texto con `pypdf`
3. Divide en chunks con `RecursiveCharacterTextSplitter`
4. Genera embeddings con `sentence-transformers`
5. Almacena en PostgreSQL con pgvector

### Chat Conversacional
1. Usuario envía pregunta → `/chat/`
2. Sistema genera embedding de la pregunta
3. Busca chunks relevantes en pgvector (similarity search)
4. Obtiene historial de conversación de la BD
5. Construye prompt con contexto + historial
6. Llama a DeepSeek API
7. Retorna respuesta + fuentes
8. Guarda intercambio en historial

## Tecnologías Utilizadas

- **FastAPI**: Framework web moderno y rápido
- **LangChain**: Framework para aplicaciones LLM
- **PostgreSQL + pgvector**: Base de datos vectorial
- **DeepSeek API**: Modelo de lenguaje para generación
- **Sentence Transformers**: Embeddings locales
- **Pydantic**: Validación de datos
- **Loguru**: Logging mejorado

## Solución de Problemas

### Error: "No module named 'psycopg'"
```bash
pip install psycopg[binary]
```

### Error: "extension 'vector' does not exist"
Instala pgvector en PostgreSQL:
```bash
# Ubuntu
sudo apt install postgresql-14-pgvector

# Mac
brew install pgvector
```

### Error de conexión a DeepSeek
Verifica que tu API key sea válida en `.env`:
```bash
DEEPSEEK_API_KEY=sk-...
```

### PDFs no se procesan
- Verifica que el PDF contenga texto (no imágenes escaneadas)
- Revisa el tamaño (máximo configurado en `MAX_FILE_SIZE_MB`)
- Mira los logs en `logs/app_YYYY-MM-DD.log`

---

## Logs y Monitoreo

Los logs se guardan automáticamente con rotación diaria:
- **Consola**: Nivel INFO (mensajes importantes)
- **Archivo**: `logs/app_YYYY-MM-DD.log` (nivel DEBUG, todos los detalles)

**Ejemplo de log**:
```
2025-01-06 10:15:23 | INFO | Chat request: user=docente01, message='¿Qué ejercicios recomiendas?...'
2025-01-06 10:15:23 | DEBUG | Encontrados 3 chunks con scores: [0.89, 0.85, 0.82]
2025-01-06 10:15:25 | INFO | Chat completado en 2.3s, tokens=350
```

---

## Recursos Adicionales

- **[Documentación de LangChain](https://python.langchain.com/)**: Framework para aplicaciones LLM
- **[DeepSeek API](https://platform.deepseek.com/)**: Documentación del modelo de lenguaje
- **[pgvector](https://github.com/pgvector/pgvector)**: Extensión PostgreSQL para vectores
- **[FastAPI](https://fastapi.tiangolo.com/)**: Framework web moderno
- **[sentence-transformers](https://www.sbert.net/)**: Modelos de embeddings
- **[Pydantic](https://docs.pydantic.dev/)**: Validación de datos

---

## Casos de Uso Pedagógicos

### 1. Preparación de Clases
Un docente sube guías didácticas y pregunta:
- "Dame ideas de actividades para introducir el tema de fracciones"
- "¿Qué ejercicios prácticos puedo usar para evaluar comprensión lectora?"

### 2. Generación de Material
Extracción automática de:
- Ejercicios clasificados por dificultad
- Materiales necesarios para cada lección
- Consejos pedagógicos aplicables

### 3. Consulta Rápida
Durante la clase, consulta en streaming:
- "¿Cómo explico el ciclo del agua de forma visual?"
- "Dame 3 estrategias para motivar estudiantes desmotivados"

### 4. Organización de Recursos
- Búsqueda por tipo de contenido (consejos, ejercicios, materiales)
- Referencias exactas de página para citas
- Historial de consultas por sesión temática

---

**Desarrollado para el curso de Minería de Datos - Universidad**
