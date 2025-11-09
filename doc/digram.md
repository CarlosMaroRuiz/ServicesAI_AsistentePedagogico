
# Arquitectura General del Sistema RAG + ML

El siguiente diagrama representa los tres flujos principales del sistema, que integran los servicios de **procesamiento de documentos**, **recuperación aumentada (RAG)** y **análisis mediante Machine Learning**.

![Architecture Diagram](../resources/architecture_diagram.png)

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

---

# Diagramas Mermaid

## 1. Arquitectura General del Sistema

```mermaid
graph TB
    subgraph "Usuario"
        U[👤 Usuario]
    end

    subgraph "services_LLM - Puerto 8000"
        LLM_API[FastAPI RAG API]

        subgraph "Endpoints"
            EP_UPLOAD["/upload"]
            EP_CHAT["/chat"]
            EP_DOCS["/documents"]
            EP_ML["/ml/*"]
        end

        subgraph "Servicios LLM"
            INGEST[Ingest Service<br/>PDF → Chunks]
            CHAT[Chat Service<br/>RAG + DeepSeek]
            QUERY[Query Service<br/>Vector Search]
            ML_CLIENT[ML Client<br/>TCP Client]
        end

        STORAGE[📁 temp/<br/>PDFs físicos]
    end

    subgraph "services_ML - Puerto 8001 + TCP 5555"
        ML_API[FastAPI ML API]
        TCP_SERVER[TCP Server<br/>Puerto 5555]

        subgraph "ML Features"
            CLUSTERING[Clustering<br/>HDBSCAN]
            TOPICS[Topic Modeling<br/>BERTopic]
            RECOMMEND[Recommendations<br/>Similarity]
            VIZ[Visualization<br/>UMAP 2D]
        end
    end

    subgraph "PostgreSQL AWS RDS"
        DB[(PostgreSQL<br/>+ pgvector)]

        subgraph "Tablas LLM"
            TB_DOCS[(documents)]
            TB_EMBED[(langchain_pg_embedding<br/>vectors 384D)]
            TB_COLL[(langchain_pg_collection)]
        end

        subgraph "Tablas ML"
            TB_CLUSTERS[(ml_clusters)]
            TB_TOPICS[(ml_topics)]
            TB_VIZ[(ml_visualizations)]
            TB_REC[(ml_recommendations)]
        end
    end

    subgraph "Servicios Externos"
        DEEPSEEK[🌐 DeepSeek API<br/>LLM]
    end

    %% Conexiones
    U --> EP_UPLOAD
    U --> EP_CHAT
    U --> EP_DOCS
    U --> EP_ML

    EP_UPLOAD --> LLM_API --> INGEST
    EP_CHAT --> LLM_API --> CHAT
    EP_DOCS --> LLM_API --> QUERY
    EP_ML --> LLM_API --> ML_CLIENT

    INGEST --> STORAGE
    INGEST --> TB_DOCS
    INGEST --> TB_EMBED
    INGEST --> TB_COLL

    CHAT --> QUERY
    QUERY --> TB_EMBED
    CHAT --> DEEPSEEK

    ML_CLIENT -.->|TCP 5555| TCP_SERVER
    TCP_SERVER --> ML_API

    ML_API --> CLUSTERING
    ML_API --> TOPICS
    ML_API --> RECOMMEND
    ML_API --> VIZ

    CLUSTERING --> TB_EMBED
    TOPICS --> TB_EMBED
    RECOMMEND --> TB_EMBED
    VIZ --> TB_EMBED

    CLUSTERING --> TB_CLUSTERS
    TOPICS --> TB_TOPICS
    VIZ --> TB_VIZ
    RECOMMEND --> TB_REC

    style U fill:#e1f5ff
    style LLM_API fill:#bbdefb
    style ML_API fill:#ffccbc
    style DB fill:#c8e6c9
    style DEEPSEEK fill:#fff9c4
    style TCP_SERVER fill:#ffccbc
```

---

## 2. Flujo 1: Upload de Documentos

```mermaid
sequenceDiagram
    actor Usuario
    participant Upload as /upload Endpoint
    participant LLM as services_LLM
    participant Ingest as Ingest Service
    participant Storage as temp/
    participant DB as PostgreSQL

    Usuario->>Upload: 1. POST /upload (PDF)
    Upload->>LLM: Recibe archivo
    LLM->>Ingest: Procesar documento

    Ingest->>Storage: 2. Guardar PDF físico
    Storage-->>Ingest: Confirmación

    Ingest->>Ingest: 3. Extraer texto (PyPDF2)
    Ingest->>Ingest: 4. Chunking (RecursiveCharacterTextSplitter)
    Ingest->>Ingest: 5. Generar embeddings (384D)

    Ingest->>DB: 6. INSERT INTO documents
    Ingest->>DB: 7. INSERT INTO langchain_pg_embedding
    Ingest->>DB: 8. INSERT INTO langchain_pg_collection

    DB-->>Ingest: Confirmación
    Ingest-->>LLM: Metadata del documento
    LLM-->>Usuario: 200 OK + document_id, chunks_created

    Note over Usuario,DB: Documento procesado y listo para consultas
```

---

## 3. Flujo 2: Chat con RAG

```mermaid
sequenceDiagram
    actor Usuario
    participant Chat as /chat Endpoint
    participant LLM as services_LLM
    participant ChatSvc as Chat Service
    participant QuerySvc as Query Service
    participant DB as PostgreSQL<br/>(langchain_pg_embedding)
    participant DeepSeek as DeepSeek API

    Usuario->>Chat: 1. POST /chat {user_id, query}
    Chat->>LLM: Recibe consulta
    LLM->>ChatSvc: Procesar chat

    ChatSvc->>QuerySvc: 2. Buscar documentos relevantes
    QuerySvc->>QuerySvc: 3. Generar embedding de la query

    QuerySvc->>DB: 4. SELECT ... ORDER BY embedding <-> query_vector
    Note over QuerySvc,DB: Búsqueda vectorial con pgvector
    DB-->>QuerySvc: 5. Top-K documentos (chunks)

    QuerySvc-->>ChatSvc: Contexto recuperado

    ChatSvc->>ChatSvc: 6. Construir prompt RAG<br/>(query + contexto)

    ChatSvc->>DeepSeek: 7. POST /chat/completions<br/>(prompt con contexto)
    DeepSeek-->>ChatSvc: 8. Respuesta generada

    ChatSvc-->>LLM: Respuesta + fuentes
    LLM-->>Usuario: 9. 200 OK {answer, sources, conversation_id}

    Note over Usuario,DeepSeek: Respuesta generada con contexto relevante
```

---

## 4. Flujo 3: Análisis ML

```mermaid
sequenceDiagram
    actor Usuario
    participant ML_EP as /ml/* Endpoints
    participant LLM as services_LLM
    participant TCP_Client as ML TCP Client
    participant TCP_Server as TCP Server<br/>(Puerto 5555)
    participant ML_API as services_ML
    participant ML_Feat as ML Features<br/>(HDBSCAN, BERTopic, UMAP)
    participant DB as PostgreSQL

    Usuario->>ML_EP: 1. POST /ml/clustering {user_id}
    ML_EP->>LLM: Recibe solicitud
    LLM->>TCP_Client: 2. Preparar request TCP

    TCP_Client->>TCP_Server: 3. TCP Request<br/>(JSON over TCP)
    Note over TCP_Client,TCP_Server: Comunicación en Puerto 5555

    TCP_Server->>ML_API: 4. Decodificar y enrutar
    ML_API->>ML_Feat: 5. Ejecutar análisis

    ML_Feat->>DB: 6. SELECT embedding FROM langchain_pg_embedding
    DB-->>ML_Feat: Embeddings (384D)

    ML_Feat->>ML_Feat: 7. UMAP (384D → 5D)<br/>para clustering
    ML_Feat->>ML_Feat: 8. HDBSCAN clustering
    ML_Feat->>ML_Feat: 9. BERTopic modeling
    ML_Feat->>ML_Feat: 10. UMAP (384D → 2D)<br/>para visualización
    ML_Feat->>ML_Feat: 11. Calcular similitud

    ML_Feat->>DB: 12. INSERT INTO ml_clusters
    ML_Feat->>DB: 13. INSERT INTO ml_topics
    ML_Feat->>DB: 14. INSERT INTO ml_visualizations
    ML_Feat->>DB: 15. INSERT INTO ml_recommendations

    DB-->>ML_Feat: Confirmación
    ML_Feat-->>ML_API: Resultados del análisis

    ML_API->>TCP_Server: 16. Codificar respuesta
    TCP_Server->>TCP_Client: 17. TCP Response<br/>(JSON)

    TCP_Client-->>LLM: Resultados ML
    LLM-->>Usuario: 18. 200 OK {clusters, topics, viz, recommendations}

    Note over Usuario,DB: Análisis ML completado y almacenado
```

---

## 5. Esquema de Base de Datos

```mermaid
erDiagram
    documents ||--o{ langchain_pg_embedding : "1:N chunks"
    documents {
        uuid id PK
        varchar user_id
        varchar filename
        varchar file_path
        int file_size
        int chunks_count
        varchar status
        timestamp created_at
    }

    langchain_pg_collection ||--o{ langchain_pg_embedding : "1:N embeddings"
    langchain_pg_collection {
        uuid uuid PK
        varchar name "user_id"
        jsonb cmetadata
    }

    langchain_pg_embedding {
        uuid id PK
        uuid collection_id FK
        uuid document "document_id"
        vector_384 embedding "384D vector"
        text content "chunk text"
        jsonb cmetadata "page, etc"
    }

    ml_clusters ||--o{ ml_document_clusters : "1:N assignments"
    ml_clusters {
        serial id PK
        varchar user_id
        int cluster_id
        varchar label
        int size
        text_array keywords
        vector_384 centroid
        timestamp created_at
    }

    ml_document_clusters {
        serial id PK
        uuid document_id
        int cluster_id FK
        float probability
        boolean is_outlier
        timestamp created_at
    }

    ml_topics ||--o{ ml_document_topics : "1:N assignments"
    ml_topics {
        serial id PK
        varchar user_id
        int topic_id
        varchar label
        text_array keywords
        int document_count
        timestamp created_at
    }

    ml_document_topics {
        serial id PK
        uuid document_id
        int topic_id FK
        float probability
        timestamp created_at
    }

    ml_visualizations {
        serial id PK
        varchar user_id
        uuid document_id
        float x "UMAP 2D"
        float y "UMAP 2D"
        int cluster_id
        timestamp created_at
    }

    ml_recommendations {
        serial id PK
        uuid document_id
        uuid recommended_document_id
        float similarity_score
        int rank
        timestamp created_at
    }

    ml_analysis_metadata {
        serial id PK
        varchar user_id
        varchar analysis_type
        varchar status
        int total_documents
        jsonb parameters
        jsonb results
        text error_message
        timestamp started_at
        timestamp completed_at
    }

    ml_temporal_patterns {
        serial id PK
        varchar user_id
        int topic_id
        date period
        int document_count
        varchar trend
        float change_percentage
        timestamp created_at
    }
```

---

## 6. Diagrama de Componentes Técnicos

```mermaid
graph LR
    subgraph "Frontend/Cliente"
        CLIENT[Cliente HTTP]
    end

    subgraph "services_LLM:8000"
        FASTAPI_LLM[FastAPI App]

        subgraph "Routers"
            R_DOCS[document_routes]
            R_CHAT[chat_routes]
        end

        subgraph "Services"
            S_INGEST[ingest_service]
            S_CHAT[chat_service]
            S_QUERY[query_service]
        end

        subgraph "Utils"
            U_EXTRACTOR[text_extractor]
            U_CHUNKER[chunker]
        end

        subgraph "DB Connection"
            POOL_LLM[ConnectionPool<br/>psycopg3]
        end

        TCP_CLIENT_LIB[TCP Client<br/>tcp_client.py]
    end

    subgraph "services_ML:8001"
        FASTAPI_ML[FastAPI App]
        TCP_SERVER_LIB[TCP Server<br/>:5555]

        subgraph "Features"
            F_CLUSTER[clustering]
            F_TOPICS[topics]
            F_RECOMMEND[recommendations]
            F_VIZ[visualization]
        end

        subgraph "ML Models"
            M_HDBSCAN[HDBSCAN]
            M_BERTOPIC[BERTopic]
            M_UMAP[UMAP]
            M_EMBEDDER[SentenceTransformer<br/>all-MiniLM-L6-v2]
        end

        subgraph "DB Connection"
            POOL_ML[ConnectionPool<br/>psycopg3]
        end
    end

    subgraph "PostgreSQL AWS RDS"
        PG[(PostgreSQL 15+<br/>pgvector extension)]
    end

    subgraph "External APIs"
        DEEPSEEK_API[DeepSeek API<br/>deepseek-chat]
    end

    %% Conexiones
    CLIENT --> FASTAPI_LLM
    FASTAPI_LLM --> R_DOCS
    FASTAPI_LLM --> R_CHAT

    R_DOCS --> S_INGEST
    R_CHAT --> S_CHAT

    S_INGEST --> U_EXTRACTOR
    S_INGEST --> U_CHUNKER
    S_INGEST --> POOL_LLM

    S_CHAT --> S_QUERY
    S_CHAT --> DEEPSEEK_API
    S_QUERY --> POOL_LLM

    POOL_LLM --> PG

    R_CHAT --> TCP_CLIENT_LIB
    TCP_CLIENT_LIB -.->|TCP 5555| TCP_SERVER_LIB

    TCP_SERVER_LIB --> FASTAPI_ML
    FASTAPI_ML --> F_CLUSTER
    FASTAPI_ML --> F_TOPICS
    FASTAPI_ML --> F_RECOMMEND
    FASTAPI_ML --> F_VIZ

    F_CLUSTER --> M_HDBSCAN
    F_CLUSTER --> M_UMAP
    F_TOPICS --> M_BERTOPIC
    F_VIZ --> M_UMAP
    F_RECOMMEND --> M_EMBEDDER

    F_CLUSTER --> POOL_ML
    F_TOPICS --> POOL_ML
    F_RECOMMEND --> POOL_ML
    F_VIZ --> POOL_ML

    POOL_ML --> PG

    style CLIENT fill:#e1f5ff
    style FASTAPI_LLM fill:#bbdefb
    style FASTAPI_ML fill:#ffccbc
    style PG fill:#c8e6c9
    style DEEPSEEK_API fill:#fff9c4
    style TCP_SERVER_LIB fill:#f8bbd0
    style TCP_CLIENT_LIB fill:#f8bbd0
```

---

## 7. Flujo de Comunicación TCP

```mermaid
sequenceDiagram
    participant LLM as services_LLM<br/>TCP Client
    participant TCP as TCP Server<br/>:5555
    participant ML as services_ML<br/>Handler

    Note over LLM,ML: Establecer conexión TCP

    LLM->>TCP: 1. Conectar a localhost:5555
    TCP-->>LLM: 2. Conexión establecida

    Note over LLM,ML: Enviar solicitud

    LLM->>LLM: 3. Crear TCPRequest<br/>{action, data, request_id}
    LLM->>LLM: 4. Serializar a JSON
    LLM->>LLM: 5. Agregar length prefix (4 bytes)

    LLM->>TCP: 6. Enviar [length][JSON]
    TCP->>TCP: 7. Leer length (4 bytes)
    TCP->>TCP: 8. Leer mensaje (length bytes)
    TCP->>TCP: 9. Decodificar JSON → TCPRequest

    TCP->>ML: 10. Enrutar a handler según action
    ML->>ML: 11. Procesar solicitud
    ML-->>TCP: 12. Retornar resultado

    Note over LLM,ML: Enviar respuesta

    TCP->>TCP: 13. Crear TCPResponse<br/>{status, result, request_id}
    TCP->>TCP: 14. Serializar a JSON
    TCP->>TCP: 15. Agregar length prefix

    TCP->>LLM: 16. Enviar [length][JSON]
    LLM->>LLM: 17. Leer length
    LLM->>LLM: 18. Leer mensaje
    LLM->>LLM: 19. Decodificar JSON → TCPResponse

    LLM->>TCP: 20. Cerrar conexión
    TCP-->>LLM: 21. Conexión cerrada

    Note over LLM,ML: Comunicación completada
```

---

## Notas Técnicas

### Tecnologías Utilizadas

**services_LLM:**
- FastAPI 0.115+
- psycopg3 (ConnectionPool)
- PyPDF2 (extracción de texto)
- LangChain (chunking y embeddings)
- sentence-transformers (all-MiniLM-L6-v2)
- DeepSeek API (generación LLM)

**services_ML:**
- FastAPI 0.115+
- asyncio (TCP Server)
- HDBSCAN (clustering)
- BERTopic (topic modeling)
- UMAP (dimensionality reduction)
- scikit-learn (similarity)

**Base de Datos:**
- PostgreSQL 15+
- pgvector extension (vector operations)
- Indexes optimizados para búsquedas vectoriales

### Configuración de Puertos

| Servicio       | Puerto | Protocolo | Descripción                          |
| -------------- | ------ | --------- | ------------------------------------ |
| services_LLM   | 8000   | HTTP      | API REST para RAG                    |
| services_ML    | 8001   | HTTP      | API REST para ML (opcional)          |
| TCP Server     | 5555   | TCP       | Comunicación inter-servicios         |
| PostgreSQL RDS | 5432   | TCP       | Base de datos (AWS RDS)              |

### Formatos de Datos

**TCPRequest:**
```json
{
  "action": "CLUSTERING|TOPICS|RECOMMENDATIONS|VISUALIZATION",
  "data": {
    "user_id": "string",
    "params": {}
  },
  "request_id": "uuid"
}
```

**TCPResponse:**
```json
{
  "status": "success|error",
  "result": {},
  "error": "string|null",
  "request_id": "uuid"
}
```
