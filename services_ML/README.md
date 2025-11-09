# Sistema de Machine Learning No Supervisado

API REST para análisis automático de recursos educativos mediante clustering, topic modeling y recomendaciones.

---

## Características Principales

### 🔍 Clustering Automático (HDBSCAN)
- Agrupamiento automático de documentos similares
- Detección de outliers (documentos que no encajan)
- Clusters jerárquicos con subcategorías
- No requiere especificar número de grupos

### 🏷️ Topic Modeling (BERTopic)
- Extracción automática de temas principales
- Etiquetas descriptivas generadas automáticamente
- Palabras clave por tema
- Actualización incremental con nuevos documentos

### 📊 Reducción Dimensional (UMAP)
- 384D → 5D para clustering eficiente
- 384D → 2D para visualizaciones interactivas
- Preserva estructura semántica

### 🎯 Sistema de Recomendaciones
- KNN + Similitud Coseno
- Búsqueda de documentos similares
- Ranking por relevancia

### 📈 Análisis Temporal
- Detección de tendencias en temas
- Temas emergentes y en declive
- Análisis de estacionalidad

---

## Stack Tecnológico

- **Framework**: FastAPI
- **Clustering**: HDBSCAN
- **Topic Modeling**: BERTopic
- **Reducción Dimensional**: UMAP
- **Embeddings**: Compartidos con services_LLM (sentence-transformers)
- **Database**: PostgreSQL + pgvector
- **Comunicación**: TCP Socket (Puerto 5555)

---

## Comunicación TCP con services_LLM

### Servidor TCP (Puerto 5555)
El servicio expone un servidor TCP que acepta comandos JSON:

```python
# Comando de ejemplo
{
    "action": "cluster_documents",
    "data": {
        "user_id": "docente01",
        "embeddings": [...],  # Array de vectores 384D
        "document_ids": [...]
    }
}
```

### Respuestas
```python
{
    "status": "success",
    "result": {
        "clusters": [...],
        "topics": [...],
        "recommendations": [...]
    }
}
```

---

## Arquitectura

```
services_ML/
├── app/
│   ├── core/
│   │   ├── config/
│   │   │   └── settings.py          # Configuración
│   │   └── tcp/
│   │       ├── server.py             # Servidor TCP
│   │       └── protocol.py           # Protocolo de mensajes
│   ├── features/
│   │   ├── clustering/
│   │   │   ├── domain/
│   │   │   │   └── entities/
│   │   │   ├── application/
│   │   │   │   ├── use_cases/
│   │   │   │   │   ├── cluster_documents.py
│   │   │   │   │   └── update_clusters.py
│   │   │   │   └── ports/
│   │   │   └── infrastructure/
│   │   │       ├── adapters/
│   │   │       │   ├── hdbscan_adapter.py
│   │   │       │   ├── umap_adapter.py
│   │   │       │   └── persistence_adapter.py
│   │   │       └── http/
│   │   │           └── controllers.py
│   │   ├── topic_modeling/
│   │   │   └── (estructura similar)
│   │   ├── recommendations/
│   │   │   └── (estructura similar)
│   │   └── temporal_analysis/
│   │       └── (estructura similar)
│   └── main.py                       # FastAPI + TCP Server
├── requirements.txt
└── .env
```

---

## Instalación

### 1. Crear entorno virtual
```bash
cd services_ML
python -m venv venv
venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
```bash
# .env
ML_SERVICE_HOST=0.0.0.0
ML_SERVICE_PORT=8001
TCP_SERVER_HOST=0.0.0.0
TCP_SERVER_PORT=5555

# Database (compartida con services_LLM)
DATABASE_URL=postgresql://raguser:RagSecurePass2024!@23.20.144.147:5432/ragdb

# ML Models
MIN_CLUSTER_SIZE=3
MIN_SAMPLES=2
UMAP_N_NEIGHBORS=15
UMAP_N_COMPONENTS=5
UMAP_METRIC=cosine
```

### 4. Ejecutar servidor
```bash
python app/main.py
```

El servicio estará disponible en:
- **REST API**: http://localhost:8001
- **TCP Server**: tcp://localhost:5555
- **Docs**: http://localhost:8001/docs

---

## API Endpoints

### Clustering

#### `POST /clustering/analyze`
Analiza y agrupa documentos automáticamente.

```json
{
  "user_id": "docente01",
  "document_ids": ["uuid-1", "uuid-2", ...]
}
```

**Respuesta:**
```json
{
  "total_documents": 50,
  "num_clusters": 4,
  "clusters": [
    {
      "cluster_id": 0,
      "label": "Matemáticas - Fracciones y Decimales",
      "size": 12,
      "document_ids": [...],
      "keywords": ["fracciones", "decimales", "operaciones"],
      "centroid": [...]
    }
  ],
  "outliers": 5
}
```

#### `GET /clustering/clusters/{user_id}`
Lista todos los clusters del usuario.

#### `GET /clustering/cluster/{cluster_id}`
Obtiene detalles de un cluster específico.

---

### Topic Modeling

#### `POST /topics/extract`
Extrae temas principales de los documentos.

```json
{
  "user_id": "docente01",
  "num_topics": 10
}
```

**Respuesta:**
```json
{
  "topics": [
    {
      "topic_id": 0,
      "label": "Estrategias de Comprensión Lectora",
      "keywords": ["lectura", "comprensión", "estrategias"],
      "document_count": 15,
      "representative_docs": [...]
    }
  ]
}
```

#### `GET /topics/trends/{user_id}`
Analiza tendencias temporales de temas.

---

### Recomendaciones

#### `POST /recommendations/similar`
Encuentra documentos similares.

```json
{
  "document_id": "uuid-123",
  "top_k": 5
}
```

**Respuesta:**
```json
{
  "recommendations": [
    {
      "document_id": "uuid-456",
      "filename": "Operaciones decimales.pdf",
      "similarity_score": 0.89,
      "cluster_id": 0
    }
  ]
}
```

---

### Visualización

#### `GET /visualization/clusters-2d/{user_id}`
Obtiene coordenadas 2D para visualización de clusters.

```json
{
  "points": [
    {
      "document_id": "uuid-1",
      "x": 12.5,
      "y": -8.3,
      "cluster_id": 0,
      "label": "Matemáticas"
    }
  ]
}
```

---

## Modelos ML Utilizados

### HDBSCAN
- **min_cluster_size**: 3 (mínimo 3 documentos por cluster)
- **min_samples**: 2
- **metric**: euclidean (post-UMAP)

### UMAP
- **n_components**: 5 (clustering), 2 (visualización)
- **n_neighbors**: 15
- **metric**: cosine

### BERTopic
- **top_n_words**: 10
- **language**: spanish
- **embeddings**: Reutiliza sentence-transformers de services_LLM

---

## Comunicación con services_LLM

### Flujo de Trabajo

```
1. Usuario sube PDF en services_LLM
   ↓
2. services_LLM genera embeddings (sentence-transformers)
   ↓
3. services_LLM guarda embeddings en PostgreSQL
   ↓
4. services_LLM envía comando TCP a services_ML:
   "Analizar nuevo documento"
   ↓
5. services_ML obtiene embeddings de PostgreSQL
   ↓
6. services_ML ejecuta clustering + topic modeling
   ↓
7. services_ML guarda resultados en PostgreSQL
   ↓
8. services_ML responde a services_LLM vía TCP
   ↓
9. services_LLM muestra clusters/temas al usuario
```

### Comandos TCP Soportados

```python
# Cluster documentos
{
    "action": "cluster_documents",
    "data": {"user_id": "docente01"}
}

# Extraer temas
{
    "action": "extract_topics",
    "data": {"user_id": "docente01", "num_topics": 10}
}

# Recomendar similares
{
    "action": "recommend_similar",
    "data": {"document_id": "uuid-123", "top_k": 5}
}

# Actualizar visualización
{
    "action": "update_visualization",
    "data": {"user_id": "docente01"}
}
```

---

## Base de Datos

### Nuevas Tablas (a crear)

```sql
-- Clusters descubiertos
CREATE TABLE ml_clusters (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    cluster_id INTEGER NOT NULL,
    label VARCHAR,
    size INTEGER,
    keywords TEXT[],
    centroid VECTOR(384),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Asignación documento-cluster
CREATE TABLE ml_document_clusters (
    id SERIAL PRIMARY KEY,
    document_id UUID NOT NULL,
    cluster_id INTEGER NOT NULL,
    probability FLOAT,
    is_outlier BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Temas descubiertos
CREATE TABLE ml_topics (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    topic_id INTEGER NOT NULL,
    label VARCHAR,
    keywords TEXT[],
    document_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Asignación documento-tema
CREATE TABLE ml_document_topics (
    id SERIAL PRIMARY KEY,
    document_id UUID NOT NULL,
    topic_id INTEGER NOT NULL,
    probability FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Visualizaciones 2D
CREATE TABLE ml_visualizations (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    document_id UUID NOT NULL,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    cluster_id INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Análisis temporal
CREATE TABLE ml_temporal_patterns (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    topic_id INTEGER,
    period DATE NOT NULL,
    document_count INTEGER,
    trend VARCHAR, -- 'increasing', 'stable', 'decreasing'
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Desarrollo

### Tests
```bash
pytest tests/
```

### Linting
```bash
ruff check app/
```

---

## Recursos

- [HDBSCAN Docs](https://hdbscan.readthedocs.io/)
- [UMAP Docs](https://umap-learn.readthedocs.io/)
- [BERTopic Docs](https://maartengr.github.io/BERTopic/)

---

**Desarrollado para el curso de Minería de Datos - Universidad**
