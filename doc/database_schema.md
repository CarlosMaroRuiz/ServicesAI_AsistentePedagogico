# Esquema de Base de Datos - Sistema RAG con ML

Este documento describe en detalle el esquema de base de datos utilizado en el sistema de Recuperación Aumentada por Generación (RAG) con Machine Learning No Supervisado.

## Información General

**Motor de Base de Datos:** PostgreSQL 15+
**Extensiones:** pgvector
**Ubicación:** AWS RDS
**Puerto:** 5432

---

## Categorías de Tablas

El esquema se divide en dos categorías principales:

1. **Tablas LLM/RAG:** Gestión de documentos, embeddings y colecciones
2. **Tablas ML:** Almacenamiento de resultados de análisis de Machine Learning

---

## 1. Tablas LLM/RAG

### 1.1 Tabla: `documents`

**Propósito:** Almacena la metadata de los documentos PDF procesados por el sistema.

**Estructura:**

| Campo         | Tipo         | Descripción                                    | Constraints       |
|---------------|--------------|------------------------------------------------|-------------------|
| id            | UUID         | Identificador único del documento              | PRIMARY KEY       |
| user_id       | VARCHAR(255) | Identificador del usuario propietario          | NOT NULL          |
| filename      | VARCHAR(500) | Nombre original del archivo                    | NOT NULL          |
| file_path     | VARCHAR(1000)| Ruta del archivo físico en temp/               | NOT NULL          |
| file_size     | INTEGER      | Tamaño del archivo en bytes                    | NOT NULL          |
| chunks_count  | INTEGER      | Número de chunks generados del documento       | DEFAULT 0         |
| status        | VARCHAR(50)  | Estado del procesamiento (processing/completed/error) | DEFAULT 'processing' |
| created_at    | TIMESTAMP    | Fecha y hora de creación                       | DEFAULT NOW()     |

**Índices:**
```sql
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_status ON documents(status);
```

**Ejemplo de registro:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "test_user_001",
  "filename": "DDD_comienzo.pdf",
  "file_path": "temp/test_user_001_DDD_comienzo.pdf",
  "file_size": 2458624,
  "chunks_count": 145,
  "status": "completed",
  "created_at": "2025-01-08 10:30:00"
}
```

---

### 1.2 Tabla: `langchain_pg_collection`

**Propósito:** Gestiona colecciones de embeddings por usuario. Cada usuario tiene su propia colección para aislar sus documentos.

**Estructura:**

| Campo     | Tipo         | Descripción                              | Constraints |
|-----------|--------------|------------------------------------------|-------------|
| uuid      | UUID         | Identificador único de la colección      | PRIMARY KEY |
| name      | VARCHAR(255) | Nombre de la colección (user_id)         | UNIQUE      |
| cmetadata | JSONB        | Metadata adicional en formato JSON       | NULL        |

**Índices:**
```sql
CREATE UNIQUE INDEX idx_collection_name ON langchain_pg_collection(name);
```

**Ejemplo de registro:**
```json
{
  "uuid": "660e8400-e29b-41d4-a716-446655440001",
  "name": "test_user_001",
  "cmetadata": {
    "created_at": "2025-01-08",
    "embedding_model": "all-MiniLM-L6-v2"
  }
}
```

---

### 1.3 Tabla: `langchain_pg_embedding`

**Propósito:** Almacena los chunks de texto extraídos de los documentos junto con sus embeddings vectoriales de 384 dimensiones.

**Estructura:**

| Campo         | Tipo         | Descripción                                    | Constraints       |
|---------------|--------------|------------------------------------------------|-------------------|
| id            | UUID         | Identificador único del embedding              | PRIMARY KEY       |
| collection_id | UUID         | Referencia a langchain_pg_collection           | FOREIGN KEY       |
| document      | UUID         | Referencia al documento original (documents.id)| NULL              |
| embedding     | VECTOR(384)  | Vector de embedding de 384 dimensiones         | NOT NULL          |
| content       | TEXT         | Contenido textual del chunk                    | NOT NULL          |
| cmetadata     | JSONB        | Metadata del chunk (página, posición, etc.)    | NULL              |

**Índices:**
```sql
CREATE INDEX idx_embedding_collection ON langchain_pg_embedding(collection_id);
CREATE INDEX idx_embedding_document ON langchain_pg_embedding(document);

-- Índice para búsqueda vectorial eficiente
CREATE INDEX idx_embedding_vector ON langchain_pg_embedding
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**Ejemplo de registro:**
```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "collection_id": "660e8400-e29b-41d4-a716-446655440001",
  "document": "550e8400-e29b-41d4-a716-446655440000",
  "embedding": [0.123, -0.456, 0.789, ..., 0.321],
  "content": "El Diseño Dirigido por el Dominio (DDD) es un enfoque de desarrollo de software que se centra en la complejidad del dominio del negocio...",
  "cmetadata": {
    "page": 5,
    "chunk_index": 12,
    "total_chunks": 145
  }
}
```

**Operaciones típicas:**

Búsqueda vectorial por similitud:
```sql
SELECT id, content, cmetadata,
       embedding <-> '[query_vector]'::vector AS distance
FROM langchain_pg_embedding
WHERE collection_id = 'user_collection_uuid'
ORDER BY embedding <-> '[query_vector]'::vector
LIMIT 5;
```

---

## 2. Tablas ML

### 2.1 Tabla: `ml_clusters`

**Propósito:** Almacena información sobre clusters descubiertos mediante HDBSCAN.

**Estructura:**

| Campo      | Tipo         | Descripción                                 | Constraints   |
|------------|--------------|---------------------------------------------|---------------|
| id         | SERIAL       | Identificador autoincremental               | PRIMARY KEY   |
| user_id    | VARCHAR(255) | Usuario propietario del análisis            | NOT NULL      |
| cluster_id | INTEGER      | ID del cluster (puede ser -1 para outliers) | NOT NULL      |
| label      | VARCHAR(500) | Etiqueta descriptiva del cluster            | NULL          |
| size       | INTEGER      | Número de documentos en el cluster          | NOT NULL      |
| keywords   | TEXT[]       | Array de palabras clave representativas     | NULL          |
| centroid   | VECTOR(384)  | Centroide del cluster en espacio vectorial  | NULL          |
| created_at | TIMESTAMP    | Fecha de creación del análisis              | DEFAULT NOW() |

**Índices:**
```sql
CREATE INDEX idx_clusters_user_id ON ml_clusters(user_id);
CREATE INDEX idx_clusters_cluster_id ON ml_clusters(cluster_id);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "user_id": "test_user_001",
  "cluster_id": 0,
  "label": "Arquitectura de Software y DDD",
  "size": 42,
  "keywords": ["dominio", "arquitectura", "bounded context", "aggregate"],
  "centroid": [0.234, -0.123, 0.567, ..., 0.890],
  "created_at": "2025-01-08 11:00:00"
}
```

---

### 2.2 Tabla: `ml_document_clusters`

**Propósito:** Relación muchos a muchos entre documentos y clusters. Un documento puede pertenecer a múltiples clusters con diferentes probabilidades.

**Estructura:**

| Campo       | Tipo      | Descripción                                  | Constraints   |
|-------------|-----------|----------------------------------------------|---------------|
| id          | SERIAL    | Identificador autoincremental                | PRIMARY KEY   |
| document_id | UUID      | Referencia al documento (documents.id)       | FOREIGN KEY   |
| cluster_id  | INTEGER   | Referencia al cluster (ml_clusters.id)       | FOREIGN KEY   |
| probability | FLOAT     | Probabilidad de pertenencia al cluster (0-1) | NULL          |
| is_outlier  | BOOLEAN   | Indica si el documento es un outlier         | DEFAULT FALSE |
| created_at  | TIMESTAMP | Fecha de asignación                          | DEFAULT NOW() |

**Índices:**
```sql
CREATE INDEX idx_doc_clusters_document ON ml_document_clusters(document_id);
CREATE INDEX idx_doc_clusters_cluster ON ml_document_clusters(cluster_id);
CREATE INDEX idx_doc_clusters_outlier ON ml_document_clusters(is_outlier);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "cluster_id": 1,
  "probability": 0.87,
  "is_outlier": false,
  "created_at": "2025-01-08 11:00:00"
}
```

---

### 2.3 Tabla: `ml_topics`

**Propósito:** Almacena temas descubiertos mediante BERTopic.

**Estructura:**

| Campo          | Tipo         | Descripción                              | Constraints   |
|----------------|--------------|------------------------------------------|---------------|
| id             | SERIAL       | Identificador autoincremental            | PRIMARY KEY   |
| user_id        | VARCHAR(255) | Usuario propietario del análisis         | NOT NULL      |
| topic_id       | INTEGER      | ID del tema                              | NOT NULL      |
| label          | VARCHAR(500) | Etiqueta descriptiva del tema            | NULL          |
| keywords       | TEXT[]       | Palabras clave más representativas       | NULL          |
| document_count | INTEGER      | Número de documentos asociados al tema   | DEFAULT 0     |
| created_at     | TIMESTAMP    | Fecha de creación del análisis           | DEFAULT NOW() |

**Índices:**
```sql
CREATE INDEX idx_topics_user_id ON ml_topics(user_id);
CREATE INDEX idx_topics_topic_id ON ml_topics(topic_id);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "user_id": "test_user_001",
  "topic_id": 0,
  "label": "Patrones de Diseño y Arquitectura",
  "keywords": ["patrón", "diseño", "arquitectura", "factory", "repository"],
  "document_count": 28,
  "created_at": "2025-01-08 11:05:00"
}
```

---

### 2.4 Tabla: `ml_document_topics`

**Propósito:** Relación entre documentos y temas descubiertos.

**Estructura:**

| Campo       | Tipo      | Descripción                              | Constraints   |
|-------------|-----------|------------------------------------------|---------------|
| id          | SERIAL    | Identificador autoincremental            | PRIMARY KEY   |
| document_id | UUID      | Referencia al documento (documents.id)   | FOREIGN KEY   |
| topic_id    | INTEGER   | Referencia al tema (ml_topics.id)        | FOREIGN KEY   |
| probability | FLOAT     | Probabilidad de pertenencia al tema      | NULL          |
| created_at  | TIMESTAMP | Fecha de asignación                      | DEFAULT NOW() |

**Índices:**
```sql
CREATE INDEX idx_doc_topics_document ON ml_document_topics(document_id);
CREATE INDEX idx_doc_topics_topic ON ml_document_topics(topic_id);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "topic_id": 1,
  "probability": 0.92,
  "created_at": "2025-01-08 11:05:00"
}
```

---

### 2.5 Tabla: `ml_visualizations`

**Propósito:** Almacena coordenadas 2D generadas por UMAP para visualización de documentos.

**Estructura:**

| Campo       | Tipo         | Descripción                             | Constraints   |
|-------------|--------------|------------------------------------------|---------------|
| id          | SERIAL       | Identificador autoincremental            | PRIMARY KEY   |
| user_id     | VARCHAR(255) | Usuario propietario                      | NOT NULL      |
| document_id | UUID         | Referencia al documento (documents.id)   | FOREIGN KEY   |
| x           | FLOAT        | Coordenada X en espacio 2D               | NOT NULL      |
| y           | FLOAT        | Coordenada Y en espacio 2D               | NOT NULL      |
| cluster_id  | INTEGER      | Cluster al que pertenece (para colores)  | NULL          |
| created_at  | TIMESTAMP    | Fecha de creación                        | DEFAULT NOW() |

**Índices:**
```sql
CREATE INDEX idx_viz_user_id ON ml_visualizations(user_id);
CREATE INDEX idx_viz_document ON ml_visualizations(document_id);
CREATE INDEX idx_viz_cluster ON ml_visualizations(cluster_id);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "user_id": "test_user_001",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "x": 12.34,
  "y": -5.67,
  "cluster_id": 0,
  "created_at": "2025-01-08 11:10:00"
}
```

**Uso típico:**
Generar gráfico de dispersión con matplotlib/plotly para visualizar distribución de documentos.

---

### 2.6 Tabla: `ml_recommendations`

**Propósito:** Almacena recomendaciones de documentos similares basadas en similitud de embeddings.

**Estructura:**

| Campo                   | Tipo      | Descripción                                  | Constraints   |
|-------------------------|-----------|----------------------------------------------|---------------|
| id                      | SERIAL    | Identificador autoincremental                | PRIMARY KEY   |
| document_id             | UUID      | Documento origen (documents.id)              | FOREIGN KEY   |
| recommended_document_id | UUID      | Documento recomendado (documents.id)         | FOREIGN KEY   |
| similarity_score        | FLOAT     | Puntuación de similitud (0-1)                | NOT NULL      |
| rank                    | INTEGER   | Ranking de la recomendación (1=más similar)  | NOT NULL      |
| created_at              | TIMESTAMP | Fecha de generación                          | DEFAULT NOW() |

**Índices:**
```sql
CREATE INDEX idx_rec_document ON ml_recommendations(document_id);
CREATE INDEX idx_rec_recommended ON ml_recommendations(recommended_document_id);
CREATE INDEX idx_rec_rank ON ml_recommendations(rank);
```

**Constraint adicional:**
```sql
ALTER TABLE ml_recommendations
ADD CONSTRAINT unique_recommendation
UNIQUE (document_id, recommended_document_id);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "recommended_document_id": "660e8400-e29b-41d4-a716-446655440003",
  "similarity_score": 0.89,
  "rank": 1,
  "created_at": "2025-01-08 11:15:00"
}
```

**Query típica:**
```sql
SELECT r.recommended_document_id, d.filename, r.similarity_score
FROM ml_recommendations r
JOIN documents d ON r.recommended_document_id = d.id
WHERE r.document_id = '550e8400-e29b-41d4-a716-446655440000'
ORDER BY r.rank
LIMIT 5;
```

---

### 2.7 Tabla: `ml_analysis_metadata`

**Propósito:** Registra metadata sobre los análisis ML ejecutados para auditoría y troubleshooting.

**Estructura:**

| Campo           | Tipo         | Descripción                                    | Constraints   |
|-----------------|--------------|------------------------------------------------|---------------|
| id              | SERIAL       | Identificador autoincremental                  | PRIMARY KEY   |
| user_id         | VARCHAR(255) | Usuario que solicitó el análisis               | NOT NULL      |
| analysis_type   | VARCHAR(100) | Tipo de análisis (clustering/topics/etc.)      | NOT NULL      |
| status          | VARCHAR(50)  | Estado (pending/running/completed/error)       | NOT NULL      |
| total_documents | INTEGER      | Total de documentos analizados                 | DEFAULT 0     |
| parameters      | JSONB        | Parámetros utilizados en el análisis           | NULL          |
| results         | JSONB        | Resumen de resultados                          | NULL          |
| error_message   | TEXT         | Mensaje de error si status=error               | NULL          |
| started_at      | TIMESTAMP    | Fecha/hora de inicio                           | DEFAULT NOW() |
| completed_at    | TIMESTAMP    | Fecha/hora de finalización                     | NULL          |

**Índices:**
```sql
CREATE INDEX idx_analysis_user ON ml_analysis_metadata(user_id);
CREATE INDEX idx_analysis_type ON ml_analysis_metadata(analysis_type);
CREATE INDEX idx_analysis_status ON ml_analysis_metadata(status);
CREATE INDEX idx_analysis_started ON ml_analysis_metadata(started_at);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "user_id": "test_user_001",
  "analysis_type": "clustering",
  "status": "completed",
  "total_documents": 50,
  "parameters": {
    "min_cluster_size": 3,
    "min_samples": 2,
    "umap_n_components": 5
  },
  "results": {
    "n_clusters": 4,
    "n_outliers": 2,
    "silhouette_score": 0.67
  },
  "error_message": null,
  "started_at": "2025-01-08 11:00:00",
  "completed_at": "2025-01-08 11:02:30"
}
```

---

### 2.8 Tabla: `ml_temporal_patterns`

**Propósito:** Analiza patrones temporales en los temas descubiertos para detectar tendencias.

**Estructura:**

| Campo             | Tipo         | Descripción                              | Constraints   |
|-------------------|--------------|------------------------------------------|---------------|
| id                | SERIAL       | Identificador autoincremental            | PRIMARY KEY   |
| user_id           | VARCHAR(255) | Usuario propietario                      | NOT NULL      |
| topic_id          | INTEGER      | Referencia al tema (ml_topics.id)        | FOREIGN KEY   |
| period            | DATE         | Periodo analizado                        | NOT NULL      |
| document_count    | INTEGER      | Número de documentos en el periodo       | DEFAULT 0     |
| trend             | VARCHAR(50)  | Tendencia (increasing/decreasing/stable) | NULL          |
| change_percentage | FLOAT        | Porcentaje de cambio respecto periodo anterior | NULL      |
| created_at        | TIMESTAMP    | Fecha de análisis                        | DEFAULT NOW() |

**Índices:**
```sql
CREATE INDEX idx_temporal_user ON ml_temporal_patterns(user_id);
CREATE INDEX idx_temporal_topic ON ml_temporal_patterns(topic_id);
CREATE INDEX idx_temporal_period ON ml_temporal_patterns(period);
```

**Ejemplo de registro:**
```json
{
  "id": 1,
  "user_id": "test_user_001",
  "topic_id": 1,
  "period": "2025-01-01",
  "document_count": 15,
  "trend": "increasing",
  "change_percentage": 25.5,
  "created_at": "2025-01-08 11:20:00"
}
```

---

## 3. Relaciones entre Tablas

### Diagrama de Relaciones

```
documents (1) -----> (N) langchain_pg_embedding
    |                         |
    |                         |
    |                    (N) langchain_pg_collection (1)
    |
    +---> (N) ml_document_clusters (N) <---- (1) ml_clusters
    |
    +---> (N) ml_document_topics (N) <---- (1) ml_topics
    |
    +---> (1) ml_visualizations
    |
    +---> (1) ml_recommendations (source)
    |
    +---> (1) ml_recommendations (target)

ml_topics (1) -----> (N) ml_temporal_patterns
```

### Relaciones Clave

1. **documents → langchain_pg_embedding**: Un documento tiene múltiples chunks (embeddings)
2. **langchain_pg_collection → langchain_pg_embedding**: Una colección agrupa todos los embeddings de un usuario
3. **documents → ml_document_clusters**: Relación N:M a través de ml_document_clusters
4. **documents → ml_document_topics**: Relación N:M a través de ml_document_topics
5. **documents → ml_visualizations**: Cada documento tiene una coordenada 2D para visualización
6. **documents → ml_recommendations**: Un documento puede tener múltiples recomendaciones

---

## 4. Extensiones de PostgreSQL

### 4.1 pgvector

**Instalación:**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Funcionalidad:**
- Almacenamiento eficiente de vectores de alta dimensión (384D)
- Operadores de distancia: `<->` (cosine), `<#>` (inner product), `<=>` (L2)
- Índices especializados: IVFFlat, HNSW

**Operadores utilizados:**

| Operador | Descripción            | Uso típico           |
|----------|------------------------|----------------------|
| `<->`    | Distancia coseno       | Búsqueda semántica   |
| `<#>`    | Producto interno       | Similitud vectorial  |
| `<=>`    | Distancia euclidiana   | Clustering espacial  |

**Ejemplo de búsqueda:**
```sql
SELECT content, embedding <-> '[0.1, 0.2, ..., 0.3]'::vector AS distance
FROM langchain_pg_embedding
ORDER BY distance
LIMIT 5;
```

---

## 5. Estrategias de Indexación

### 5.1 Índices Vectoriales

**IVFFlat (usado actualmente):**
```sql
CREATE INDEX idx_embedding_vector ON langchain_pg_embedding
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

- **Ventajas**: Menor uso de memoria, búsquedas rápidas
- **Desventajas**: Requiere VACUUM ANALYZE después de inserciones masivas
- **Configuración**: `lists = sqrt(total_rows)`

**Alternativa HNSW (para datasets grandes):**
```sql
CREATE INDEX idx_embedding_hnsw ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### 5.2 Índices B-Tree

Para campos de búsqueda frecuente:
```sql
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_created_at ON documents(created_at);
```

### 5.3 Índices GIN para JSONB

Para búsquedas en metadata:
```sql
CREATE INDEX idx_embedding_metadata ON langchain_pg_embedding
USING GIN (cmetadata);
```

---

## 6. Mantenimiento y Optimización

### 6.1 VACUUM y ANALYZE

Ejecutar periódicamente para mantener rendimiento:
```sql
VACUUM ANALYZE langchain_pg_embedding;
VACUUM ANALYZE documents;
```

### 6.2 Limpieza de Datos Antiguos

Script para eliminar análisis antiguos:
```sql
DELETE FROM ml_analysis_metadata
WHERE completed_at < NOW() - INTERVAL '30 days'
AND status = 'completed';
```

### 6.3 Particionamiento (futuro)

Para tablas grandes, considerar particionamiento por user_id o fecha:
```sql
CREATE TABLE ml_analysis_metadata_2025_01
PARTITION OF ml_analysis_metadata
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

---

## 7. Consultas Comunes

### 7.1 Obtener documentos de un usuario con sus chunks

```sql
SELECT
    d.id,
    d.filename,
    COUNT(e.id) as chunk_count,
    d.created_at
FROM documents d
LEFT JOIN langchain_pg_embedding e ON e.document = d.id
WHERE d.user_id = 'test_user_001'
GROUP BY d.id, d.filename, d.created_at
ORDER BY d.created_at DESC;
```

### 7.2 Buscar documentos por similitud semántica

```sql
SELECT
    e.content,
    d.filename,
    e.embedding <-> '[query_vector]'::vector AS distance
FROM langchain_pg_embedding e
JOIN documents d ON e.document = d.id
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'test_user_001'
ORDER BY distance
LIMIT 5;
```

### 7.3 Obtener clusters con sus documentos

```sql
SELECT
    c.cluster_id,
    c.label,
    c.size,
    c.keywords,
    d.filename,
    dc.probability
FROM ml_clusters c
JOIN ml_document_clusters dc ON dc.cluster_id = c.id
JOIN documents d ON dc.document_id = d.id
WHERE c.user_id = 'test_user_001'
AND dc.is_outlier = false
ORDER BY c.cluster_id, dc.probability DESC;
```

### 7.4 Obtener temas con mayor cantidad de documentos

```sql
SELECT
    t.topic_id,
    t.label,
    t.keywords,
    t.document_count
FROM ml_topics t
WHERE t.user_id = 'test_user_001'
ORDER BY t.document_count DESC
LIMIT 10;
```

### 7.5 Obtener recomendaciones para un documento

```sql
SELECT
    d.filename,
    r.similarity_score,
    r.rank
FROM ml_recommendations r
JOIN documents d ON r.recommended_document_id = d.id
WHERE r.document_id = '550e8400-e29b-41d4-a716-446655440000'
ORDER BY r.rank;
```

---

## 8. Consideraciones de Seguridad

### 8.1 Aislamiento por Usuario

Todas las consultas deben filtrar por `user_id` para garantizar aislamiento:
```sql
WHERE user_id = :current_user_id
```

### 8.2 Prepared Statements

Usar siempre prepared statements para prevenir SQL injection:
```python
cursor.execute(
    "SELECT * FROM documents WHERE user_id = %s",
    (user_id,)
)
```

### 8.3 Connection Pooling

Configuración segura del pool:
```python
ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=2,
    max_size=10,
    timeout=30,
    max_idle=300
)
```

---

## 9. Backup y Recuperación

### 9.1 Backup Completo

```bash
pg_dump -h rds-endpoint -U username -d database_name > backup.sql
```

### 9.2 Backup Solo de Esquema

```bash
pg_dump -h rds-endpoint -U username -d database_name --schema-only > schema.sql
```

### 9.3 Backup de Tablas Específicas

```bash
pg_dump -h rds-endpoint -U username -d database_name \
  -t documents -t langchain_pg_embedding > documents_backup.sql
```

### 9.4 Restauración

```bash
psql -h rds-endpoint -U username -d database_name < backup.sql
```

---

## 10. Monitoreo y Métricas

### 10.1 Tamaño de Tablas

```sql
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### 10.2 Estadísticas de Uso de Índices

```sql
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### 10.3 Queries Lentas

```sql
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

---

## Resumen

Este esquema de base de datos está diseñado para:

1. **Eficiencia**: Índices vectoriales optimizados para búsquedas semánticas rápidas
2. **Escalabilidad**: Particionamiento por usuario y capacidad de crecimiento horizontal
3. **Trazabilidad**: Tablas de metadata para auditoría y debugging
4. **Flexibilidad**: JSONB para metadata extensible sin cambios de esquema
5. **Aislamiento**: Separación estricta de datos por usuario

El uso de pgvector permite realizar búsquedas semánticas eficientes sobre millones de embeddings, mientras que las tablas ML almacenan resultados de análisis avanzados para consulta rápida sin necesidad de recomputación.
