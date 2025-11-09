# Sistema RAG con Machine Learning No Supervisado

Sistema completo de **Retrieval-Augmented Generation (RAG)** con **Machine Learning No Supervisado** para análisis y gestión inteligente de recursos pedagógicos.

## Descripción

Este proyecto implementa un sistema avanzado de gestión documental que combina:

- **RAG (Retrieval-Augmented Generation)**: Chat conversacional basado en documentos con contexto semántico
- **Machine Learning No Supervisado**: Clustering, topic modeling, recomendaciones y visualización
- **Arquitectura de Microservicios**: Separación de servicios LLM y ML con comunicación TCP
- **Base de Datos Vectorial**: PostgreSQL con pgvector para búsquedas semánticas eficientes

## Características Principales

### 1. Procesamiento de Documentos
- Carga y procesamiento de archivos PDF
- Extracción de texto con PyPDF2
- Chunking inteligente de contenido
- Generación de embeddings con sentence-transformers (384D)
- Almacenamiento vectorial en PostgreSQL

### 2. Chat Conversacional (RAG)
- Búsqueda semántica sobre documentos procesados
- Generación de respuestas contextualizadas con DeepSeek API
- Historial de conversaciones
- Referencias a fuentes utilizadas

### 3. Análisis ML No Supervisado
- **Clustering**: Agrupación automática de documentos similares (HDBSCAN)
- **Topic Modeling**: Descubrimiento de temas principales (BERTopic)
- **Recommendations**: Sistema de recomendación basado en similitud
- **Visualization**: Proyección 2D de documentos con UMAP

## Arquitectura

El sistema está compuesto por dos servicios principales:

- **services_LLM** (Puerto 8000): Servicio de RAG y orquestación
- **services_ML** (Puerto 8001 + TCP 5555): Servicio de análisis ML

Ambos servicios se conectan a una base de datos PostgreSQL en AWS RDS con extensión pgvector.

![Diagrama de Arquitectura](resources/architecture_diagram.png)

## Tecnologías Utilizadas

### Backend
- **FastAPI**: Framework web asíncrono para APIs REST
- **psycopg3**: Driver PostgreSQL con soporte para connection pooling
- **LangChain**: Framework para aplicaciones LLM
- **sentence-transformers**: Generación de embeddings (all-MiniLM-L6-v2)

### Machine Learning
- **HDBSCAN**: Clustering jerárquico basado en densidad
- **BERTopic**: Topic modeling con transformers
- **UMAP**: Reducción de dimensionalidad
- **scikit-learn**: Cálculo de similitud y métricas

### Base de Datos
- **PostgreSQL 15+**: Base de datos relacional
- **pgvector**: Extensión para operaciones vectoriales eficientes

### LLM
- **DeepSeek API**: Modelo de lenguaje para generación de respuestas

## Estructura del Proyecto

```
llm_api/
├── services_LLM/           # Servicio de RAG
│   ├── app/
│   │   ├── routes/         # Endpoints REST
│   │   ├── services/       # Lógica de negocio
│   │   ├── models/         # Modelos Pydantic
│   │   ├── db/             # Conexión a BD
│   │   └── utils/          # Utilidades
│   └── main.py
├── services_ML/            # Servicio de ML
│   ├── app/
│   │   ├── features/       # Features ML (clustering, topics, etc.)
│   │   ├── core/           # TCP server, database
│   │   └── shared/         # Código compartido
│   └── main.py
├── doc/                    # Documentación técnica
│   ├── architecture.md     # Arquitectura y flujos
│   ├── database_schema.md  # Esquema de base de datos
│   ├── digram.md          # Diagramas Mermaid
│   └── comands.md         # Comandos útiles
├── test_documents/         # PDFs de prueba
├── temp/                   # Almacenamiento temporal de PDFs
└── README.md              # Este archivo
```

## Instalación

### Requisitos Previos

- Python 3.11+
- PostgreSQL 15+ con extensión pgvector
- Cuenta en DeepSeek API

### 1. Clonar el Repositorio

```bash
git clone https://github.com/CarlosMaroRuiz/ServicesAI_AsistentePedagogico.git
cd ServicesAI_AsistentePedagogico
```

### 2. Instalar Dependencias

#### Servicio LLM
```bash
cd services_LLM
pip install -r requirements.txt
```

#### Servicio ML
```bash
cd services_ML
pip install -r requirements.txt
```

### 3. Configurar Variables de Entorno

Crear archivo `.env` en la raíz del proyecto:

```env
# PostgreSQL Configuration
DATABASE_URL=postgresql://user:password@host:5432/database
DB_MIN_CONNECTIONS=2
DB_MAX_CONNECTIONS=10

# DeepSeek API
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_MODEL=deepseek-chat

# TCP Configuration
ML_TCP_HOST=localhost
ML_TCP_PORT=5555

# Server Ports
LLM_PORT=8000
ML_PORT=8001
```

### 4. Inicializar Base de Datos

```bash
# Ejecutar script de inicialización de tablas ML
python init_ml_tables.py
```

### 5. Ejecutar Servicios

#### Terminal 1: Servicio ML
```bash
cd services_ML
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

#### Terminal 2: Servicio LLM
```bash
cd services_LLM
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Uso

### Acceder a la Documentación Interactiva

- **Servicio LLM**: http://localhost:8000/docs
- **Servicio ML**: http://localhost:8001/docs

Ambos servicios usan Scalar para documentación interactiva de la API.

### Endpoints Principales

#### Gestión de Documentos
- `POST /upload` - Subir documento PDF
- `GET /documents/{user_id}` - Listar documentos del usuario
- `DELETE /documents/{document_id}` - Eliminar documento

#### Chat Conversacional
- `POST /chat` - Enviar mensaje al chat RAG
- `GET /chat/history/{user_id}` - Obtener historial de conversaciones

#### Análisis ML
- `POST /ml/clustering` - Ejecutar clustering de documentos
- `POST /ml/topics` - Descubrir temas principales
- `POST /ml/recommendations/{document_id}` - Obtener recomendaciones
- `POST /ml/visualization` - Generar visualización 2D

### Ejemplo de Uso con cURL

#### 1. Subir Documento
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf" \
  -F "user_id=user_001"
```

#### 2. Chat con RAG
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "query": "¿Qué es el Diseño Dirigido por el Dominio?"
  }'
```

#### 3. Análisis de Clustering
```bash
curl -X POST "http://localhost:8000/ml/clustering" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001"
  }'
```

## Documentación Técnica

Para información detallada sobre la arquitectura, base de datos y flujos del sistema, consulta la carpeta `doc/`:

### Documentación Disponible

| Documento | Descripción |
|-----------|-------------|
| [architecture.md](doc/architecture.md) | Arquitectura general del sistema y flujos principales |
| [database_schema.md](doc/database_schema.md) | Esquema completo de base de datos con ejemplos y queries |
| [digram.md](doc/digram.md) | Diagramas Mermaid de arquitectura, secuencia y BD |
| [comands.md](doc/comands.md) | Comandos útiles para desarrollo y mantenimiento |

### Temas Cubiertos en la Documentación

#### Arquitectura
- Componentes del sistema
- Flujo de upload de documentos
- Flujo de chat con RAG
- Flujo de análisis ML
- Comunicación TCP entre servicios
- Diagramas técnicos

#### Base de Datos
- Tablas LLM/RAG: documents, embeddings, collections
- Tablas ML: clusters, topics, visualizations, recommendations
- Índices y optimizaciones
- Extensión pgvector
- Consultas SQL comunes
- Estrategias de backup

#### Diagramas
- Arquitectura general (Mermaid)
- Diagramas de secuencia
- Entity-Relationship Diagram
- Diagrama de componentes
- Flujo de comunicación TCP

## Configuración Avanzada

### Connection Pool PostgreSQL

```python
ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=2,
    max_size=10,
    timeout=30,
    max_idle=300,  # 5 minutos
    reconnect_timeout=10
)
```

### Parámetros de Clustering

```python
{
    "min_cluster_size": 3,      # Mínimo de documentos por cluster
    "min_samples": 2,           # Mínimo de muestras en vecindario
    "umap_n_components": 5      # Dimensiones para clustering
}
```

### Parámetros de Topic Modeling

```python
{
    "min_topic_size": 3,        # Mínimo de documentos por tema
    "n_gram_range": (1, 2),     # Unigramas y bigramas
    "top_n_words": 10           # Palabras clave por tema
}
```

## Funcionamiento de los Módulos

### Módulo 1: Procesamiento de Documentos

El procesamiento de documentos sigue un flujo automatizado que transforma PDFs en datos consultables:

#### Flujo de Procesamiento

1. **Upload del Documento**
   - El usuario envía un PDF mediante `POST /upload`
   - El archivo se guarda físicamente en `temp/{user_id}_{filename}.pdf`
   - Se registra en la tabla `documents` con metadata inicial

2. **Extracción de Texto**
   - PyPDF2 extrae el texto página por página
   - Se preserva información de página y posición
   - Manejo de errores para PDFs con problemas de codificación

3. **Chunking Inteligente**
   - RecursiveCharacterTextSplitter divide el texto en fragmentos
   - Tamaño configurable: 1000 caracteres por chunk
   - Overlap de 200 caracteres para mantener contexto
   - Preserva coherencia semántica en los cortes

4. **Generación de Embeddings**
   - Modelo: sentence-transformers/all-MiniLM-L6-v2
   - Genera vectores de 384 dimensiones por chunk
   - Optimizado para búsqueda semántica en español
   - Procesamiento en batch para eficiencia

5. **Almacenamiento Vectorial**
   - Embeddings se guardan en `langchain_pg_embedding`
   - Índice IVFFlat para búsquedas rápidas
   - Metadata JSON incluye página, posición, documento origen
   - Colección por usuario para aislamiento de datos

#### Archivos Involucrados
- `services_LLM/app/routes/document_routes.py` - Endpoints de upload
- `services_LLM/app/services/ingest_service.py` - Lógica de procesamiento
- `services_LLM/app/utils/text_extractor.py` - Extracción de texto
- `services_LLM/app/utils/chunker.py` - División en chunks

---

### Módulo 2: Chat Conversacional (RAG)

Sistema de Retrieval-Augmented Generation que combina búsqueda vectorial con generación de respuestas.

#### Arquitectura RAG

1. **Recepción de Query**
   - Usuario envía pregunta en lenguaje natural
   - Se valida usuario y se recupera historial si existe
   - Se genera embedding de la pregunta (384D)

2. **Búsqueda Vectorial**
   - Query contra `langchain_pg_embedding` usando pgvector
   - Operador de distancia coseno `<->`
   - Recupera Top-K chunks más relevantes (K=5 por defecto)
   - Filtra por colección del usuario

3. **Construcción del Contexto**
   - Chunks recuperados se ordenan por relevancia
   - Se agregan metadata: documento, página, score
   - Se formatea contexto con estructura clara
   - Límite de tokens para evitar overflow en LLM

4. **Generación con LLM**
   - API: DeepSeek (modelo deepseek-chat)
   - Prompt incluye: contexto + pregunta + instrucciones
   - Temperatura: 0.3 para respuestas consistentes
   - Max tokens: 1000 para respuestas concisas

5. **Post-procesamiento**
   - Extracción de fuentes utilizadas
   - Formateo de referencias a documentos
   - Almacenamiento de conversación en historial
   - Devolución de respuesta + metadata

#### Prompts del Sistema

```
Sistema: Eres un asistente pedagógico experto. Responde basándote
exclusivamente en el contexto proporcionado. Si la información no está
en el contexto, indícalo claramente.

Contexto:
[Chunks recuperados con metadata]

Pregunta del usuario:
[Query original]
```

#### Ejemplo de Flujo Completo

```
Usuario: "¿Qué ejercicios recomienda el documento para enseñar fracciones?"
    ↓
1. Embedding de query generado (384D)
    ↓
2. Búsqueda vectorial retorna 5 chunks relevantes de documento "matematicas.pdf"
    ↓
3. Contexto construido con contenido de páginas 12, 15, 23
    ↓
4. DeepSeek genera respuesta basada en contexto
    ↓
5. Respuesta: "Según el documento, se recomiendan 3 tipos de ejercicios:
   - Comparación de fracciones con material concreto (pág. 12)
   - Suma de fracciones homogéneas con dibujos (pág. 15)
   - Problemas contextualizados de la vida cotidiana (pág. 23)"
```

#### Archivos Involucrados
- `services_LLM/app/routes/chat_routes.py` - Endpoints de chat
- `services_LLM/app/services/chat_service.py` - Orquestación RAG
- `services_LLM/app/services/query_service.py` - Búsqueda vectorial
- `services_LLM/app/models/chat_model.py` - Modelos Pydantic

---

### Módulo 3: Clustering de Documentos

Agrupación automática de documentos similares usando HDBSCAN (Hierarchical Density-Based Spatial Clustering).

#### Pipeline de Clustering

1. **Recuperación de Embeddings**
   - Se obtienen todos los embeddings del usuario
   - Agregación por documento (promedio de chunks)
   - Resultado: 1 vector de 384D por documento

2. **Reducción Dimensional (UMAP)**
   - De 384D a 5D para clustering
   - Parámetros: n_neighbors=15, min_dist=0.1
   - Preserva estructura local y global
   - Métrica: cosine distance

3. **Clustering con HDBSCAN**
   - min_cluster_size=3 (mínimo 3 docs por cluster)
   - min_samples=2 (densidad mínima)
   - Permite outliers (etiqueta -1)
   - Genera probabilidades de pertenencia

4. **Generación de Labels**
   - Se extraen keywords más frecuentes por cluster
   - TF-IDF sobre documentos del cluster
   - Top 5 términos más representativos
   - Label automático: "Cluster: keyword1, keyword2, ..."

5. **Cálculo de Centroides**
   - Promedio de vectores del cluster
   - Se almacena en espacio original (384D)
   - Útil para clasificar nuevos documentos

6. **Almacenamiento**
   - `ml_clusters`: Info general del cluster
   - `ml_document_clusters`: Asignaciones documento-cluster
   - `ml_analysis_metadata`: Métricas del proceso

#### Métricas Calculadas

- Número de clusters descubiertos
- Número de outliers
- Silhouette score (calidad del clustering)
- Tamaño de cada cluster
- Distribución de probabilidades

#### Archivos Involucrados
- `services_ML/app/features/clustering/application/use_cases/perform_clustering.py`
- `services_ML/app/features/clustering/domain/services/clustering_service.py`
- `services_ML/app/features/clustering/infrastructure/adapters/persistence_adapter.py`

---

### Módulo 4: Topic Modeling

Descubrimiento de temas principales en los documentos usando BERTopic.

#### Pipeline de Topic Modeling

1. **Preparación de Documentos**
   - Se recupera el texto completo de cada documento
   - Se unen todos los chunks del mismo documento
   - Se limpian caracteres especiales

2. **Embeddings Existentes**
   - BERTopic utiliza los embeddings ya generados (384D)
   - No es necesario re-calcular
   - Aprovecha sentence-transformers previo

3. **Reducción Dimensional**
   - UMAP reduce de 384D a 5D
   - Parámetros optimizados para topic modeling
   - n_neighbors=10, n_components=5

4. **Clustering de Temas**
   - HDBSCAN agrupa documentos en temas
   - min_topic_size=3 (mínimo 3 docs por tema)
   - Temas coherentes semánticamente

5. **Extracción de Keywords**
   - CountVectorizer extrae n-gramas (1-2 palabras)
   - c-TF-IDF identifica términos representativos
   - Top 10 palabras por tema
   - Filtra stopwords en español

6. **Generación de Labels**
   - Label automático con top 3 keywords
   - Ejemplo: "educación, pedagogía, metodología"
   - Almacenado en `ml_topics`

#### Diferencia con Clustering

| Clustering | Topic Modeling |
|------------|----------------|
| Agrupa documentos similares | Descubre temas semánticos |
| Basado en distancia vectorial | Basado en co-ocurrencia de palabras |
| HDBSCAN directo | HDBSCAN + c-TF-IDF |
| Output: grupos de docs | Output: temas con keywords |

#### Archivos Involucrados
- `services_ML/app/features/topics/application/use_cases/discover_topics.py`
- `services_ML/app/features/topics/domain/services/topic_service.py`
- `services_ML/app/features/topics/infrastructure/adapters/persistence_adapter.py`

---

### Módulo 5: Sistema de Recomendaciones

Genera recomendaciones de documentos relacionados basadas en similitud semántica.

#### Algoritmo de Recomendación

1. **Documento Base**
   - Usuario solicita recomendaciones para documento X
   - Se recupera embedding del documento (promedio de chunks)

2. **Cálculo de Similitud**
   - Similitud coseno contra todos los demás documentos
   - Formula: `cos_sim = dot(vec_A, vec_B) / (norm(A) * norm(B))`
   - Rango: [0, 1] (1 = idénticos, 0 = ortogonales)

3. **Ranking**
   - Se ordenan documentos por similitud descendente
   - Se excluye el documento base
   - Top-N recomendaciones (N=5 por defecto)

4. **Filtrado**
   - Threshold mínimo: 0.3 de similitud
   - Solo documentos del mismo usuario
   - Documentos activos (no eliminados)

5. **Enriquecimiento**
   - Se agregan metadata: filename, tamaño, fecha
   - Razón de recomendación (cluster compartido, tema común)
   - Score normalizado a porcentaje

6. **Almacenamiento**
   - Tabla `ml_recommendations`
   - Incluye: documento origen, recomendado, score, rank
   - Actualización periódica cuando hay nuevos documentos

#### Estrategias Adicionales

**Collaborative Filtering (futuro)**
- Basado en documentos que otros usuarios similares consultaron
- Requiere más usuarios para ser efectivo

**Content-Based Enhancement**
- Boost de similitud si comparten cluster
- Boost si comparten tema principal
- Penalización por diferencia de longitud extrema

#### Archivos Involucrados
- `services_ML/app/features/recommendations/application/use_cases/recommend_similar.py`
- `services_ML/app/features/recommendations/domain/services/recommendation_service.py`
- `services_ML/app/features/recommendations/infrastructure/adapters/persistence_adapter.py`

---

### Módulo 6: Visualización 2D

Proyección de documentos en espacio 2D para visualización interactiva.

#### Pipeline de Visualización

1. **Reducción Dimensional**
   - UMAP reduce de 384D a 2D
   - n_neighbors=15, min_dist=0.1
   - Preserva estructura de agrupamientos
   - Métrica: cosine distance

2. **Asignación de Colores**
   - Si existe clustering: color por cluster
   - Si no: color uniforme o por tema
   - Paleta de colores distintiva
   - Outliers en color especial (gris)

3. **Cálculo de Coordenadas**
   - Cada documento obtiene (x, y)
   - Rango normalizado para visualización web
   - Se almacena en `ml_visualizations`

4. **Generación de Datos**
   - JSON con coordenadas + metadata
   - Incluye: id, nombre, cluster, tema, (x,y)
   - Listo para consumo en frontend

#### Ejemplo de Salida

```json
{
  "visualizations": [
    {
      "document_id": "uuid-1",
      "filename": "matematicas_primaria.pdf",
      "x": 12.34,
      "y": -5.67,
      "cluster_id": 0,
      "cluster_label": "Matemáticas Básicas",
      "color": "#FF5733"
    },
    {
      "document_id": "uuid-2",
      "filename": "geometria_avanzada.pdf",
      "x": 15.21,
      "y": -3.45,
      "cluster_id": 0,
      "cluster_label": "Matemáticas Básicas",
      "color": "#FF5733"
    }
  ]
}
```

#### Casos de Uso

- Dashboard visual de todos los documentos
- Identificación de outliers (docs únicos)
- Exploración interactiva de agrupamientos
- Detección visual de clusters faltantes

#### Archivos Involucrados
- `services_ML/app/features/visualization/application/use_cases/generate_visualization.py`
- `services_ML/app/features/visualization/domain/services/visualization_service.py`
- `services_ML/app/features/visualization/infrastructure/adapters/persistence_adapter.py`

---

### Módulo 7: Comunicación TCP

Sistema de comunicación asíncrona entre services_LLM y services_ML.

#### Arquitectura TCP

**¿Por qué TCP en lugar de HTTP?**
- Conexiones persistentes (menor latencia)
- Protocolo binario eficiente
- Control total sobre serialización
- Ideal para comunicación inter-servicios

#### Protocolo de Mensajes

**Formato**: `[4 bytes length] + [JSON payload]`

```
┌─────────────┬────────────────────────┐
│   Length    │     JSON Message       │
│  (4 bytes)  │    (variable size)     │
└─────────────┴────────────────────────┘
```

**TCPRequest:**
```python
{
    "action": "CLUSTERING",  # o TOPICS, RECOMMENDATIONS, VISUALIZATION
    "data": {
        "user_id": "test_user_001",
        "params": {
            "min_cluster_size": 3,
            "min_samples": 2
        }
    },
    "request_id": "uuid-request"
}
```

**TCPResponse:**
```python
{
    "status": "success",  # o "error"
    "result": {
        "clusters": [...],
        "total_clusters": 4,
        "outliers": 2
    },
    "error": null,
    "request_id": "uuid-request"
}
```

#### Flujo de Comunicación

1. **Cliente (services_LLM)**
   ```python
   # 1. Conectar
   sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   sock.connect(('localhost', 5555))

   # 2. Serializar request
   message = json.dumps(request_dict).encode('utf-8')
   length = struct.pack('!I', len(message))

   # 3. Enviar
   sock.sendall(length + message)

   # 4. Recibir respuesta
   length_bytes = sock.recv(4)
   length = struct.unpack('!I', length_bytes)[0]
   response_bytes = sock.recv(length)

   # 5. Cerrar
   sock.close()
   ```

2. **Servidor (services_ML)**
   ```python
   # 1. Escuchar en puerto 5555
   server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   server.bind(('0.0.0.0', 5555))
   server.listen(5)

   # 2. Aceptar conexión
   while True:
       client, addr = server.accept()

       # 3. Leer length
       length = struct.unpack('!I', client.recv(4))[0]

       # 4. Leer mensaje
       message = client.recv(length)
       request = json.loads(message)

       # 5. Procesar
       result = handle_request(request)

       # 6. Enviar respuesta
       response = json.dumps(result).encode('utf-8')
       client.sendall(struct.pack('!I', len(response)) + response)

       # 7. Cerrar conexión
       client.close()
   ```

#### Manejo de Errores

- **Timeout**: 30 segundos por request
- **Retry**: 3 intentos con backoff exponencial
- **Fallback**: Respuesta de error si TCP falla

#### Archivos Involucrados
- `services_LLM/app/services/ml_client.py` - Cliente TCP
- `services_ML/app/core/tcp/server.py` - Servidor TCP
- `services_ML/app/core/tcp/protocol.py` - Protocolo de mensajes
- `services_ML/app/core/tcp/handlers.py` - Handlers por acción

---

## Integración de Módulos

### Caso de Uso Completo: Desde Upload hasta Recomendaciones

```
1. Usuario sube "metodologia_activa.pdf"
   ├─ Módulo 1: Procesamiento
   │  ├─ Extracción: 50 páginas
   │  ├─ Chunking: 145 fragmentos
   │  ├─ Embeddings: 145 vectores de 384D
   │  └─ Storage: PostgreSQL

2. Usuario pregunta "¿Qué dice sobre aprendizaje colaborativo?"
   ├─ Módulo 2: Chat RAG
   │  ├─ Embedding de query
   │  ├─ Búsqueda vectorial: 5 chunks relevantes
   │  ├─ Contexto: páginas 12, 23, 34, 45, 48
   │  ├─ DeepSeek genera respuesta
   │  └─ Respuesta + referencias

3. Sistema ejecuta análisis ML (automático o manual)
   ├─ Módulo 3: Clustering
   │  ├─ Agrupa en 4 clusters
   │  ├─ "metodologia_activa.pdf" → Cluster 1: "Pedagogía Moderna"
   │  └─ Storage: ml_clusters, ml_document_clusters
   │
   ├─ Módulo 4: Topic Modeling
   │  ├─ Descubre 6 temas
   │  ├─ "metodologia_activa.pdf" → Tema 2: "aprendizaje, colaborativo, activo"
   │  └─ Storage: ml_topics, ml_document_topics
   │
   ├─ Módulo 5: Recomendaciones
   │  ├─ Calcula similitud con otros docs
   │  ├─ Top 5 recomendados:
   │  │   1. "aprendizaje_cooperativo.pdf" (0.87)
   │  │   2. "dinamicas_grupales.pdf" (0.82)
   │  │   3. "trabajo_equipo.pdf" (0.79)
   │  │   4. "evaluacion_formativa.pdf" (0.71)
   │  │   5. "estrategias_participativas.pdf" (0.68)
   │  └─ Storage: ml_recommendations
   │
   └─ Módulo 6: Visualización
      ├─ UMAP 2D projection
      ├─ Coordenadas: (12.3, -5.6)
      ├─ Color: #FF5733 (Cluster 1)
      └─ Storage: ml_visualizations

4. Usuario consulta recomendaciones
   └─ API retorna top 5 documentos relacionados
```

### Comunicación entre Servicios

```
services_LLM (8000)                    services_ML (8001 + TCP 5555)
       │                                          │
       │  POST /ml/clustering                     │
       ├──────────────────────────────────────────►
       │                                          │
       │  TCP Request (action: CLUSTERING)        │
       ├══════════════════════════════════════════►
       │                                          ├─ Retrieve embeddings
       │                                          ├─ UMAP reduction
       │                                          ├─ HDBSCAN clustering
       │                                          ├─ Generate labels
       │                                          └─ Store results
       │  TCP Response (clusters data)            │
       ◄══════════════════════════════════════════┤
       │                                          │
       │  200 OK (formatted response)             │
       ◄──────────────────────────────────────────┤
       │                                          │
```

---

## Troubleshooting

### Error: Connection Pool Timeout

**Problema**: PostgreSQL cierra conexiones idle
**Solución**: Configurar `max_idle=300` en ConnectionPool

### Error: TCP Connection Refused

**Problema**: Servicio ML no está corriendo
**Solución**: Verificar que services_ML esté ejecutándose en puerto 5555

### Error: Insufficient Documents for ML

**Problema**: HDBSCAN requiere mínimo 3 documentos
**Solución**: Cargar más documentos antes de ejecutar clustering

### Error: pgvector Extension Missing

**Problema**: Extension pgvector no instalada
**Solución**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Repositorio

Este proyecto está disponible en GitHub:
[https://github.com/CarlosMaroRuiz/ServicesAI_AsistentePedagogico](https://github.com/CarlosMaroRuiz/ServicesAI_AsistentePedagogico)

---

Para más información técnica detallada, consulta la [documentación completa en doc/](doc/)
