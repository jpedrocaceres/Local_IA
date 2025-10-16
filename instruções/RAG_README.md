# Sistema RAG com pgvector

Sistema de Retrieval-Augmented Generation (RAG) integrado ao chat, usando PostgreSQL com pgvector para busca vetorial semântica.

## Arquitetura

```
┌─────────────────┐
│  Upload File    │
│  (FastAPI)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Extract Text   │─────▶│  Index Document  │
│  (PyPDF2, etc)  │      │  (rag_db.py)     │
└─────────────────┘      └────────┬─────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Chunk Text     │
                         │  (500 chars)    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Generate       │
                         │  Embeddings     │
                         │  (MiniLM-L6-v2) │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  PostgreSQL     │
                         │  + pgvector     │
                         └─────────────────┘

┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Embed Query    │─────▶│  Vector Search   │
└─────────────────┘      │  (Cosine Sim)    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Retrieve Top-K │
                         │  Chunks         │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Add to LLM     │
                         │  Context        │
                         └─────────────────┘
```

## Configuração do PostgreSQL com pgvector

### 1. Iniciar o container Docker

```bash
docker run -d \
  --name pgvector \
  -p 5433:5432 \
  -e POSTGRES_PASSWORD=agro123 \
  -e POSTGRES_DB=vetorial_bd \
  -v pgvector_data:/var/lib/postgresql/data \
  ankane/pgvector
```

### 2. Verificar se o container está rodando

```bash
docker ps
```

### 3. Conectar ao banco (opcional - para verificar)

```bash
docker exec -it pgvector psql -U postgres -d vetorial_bd
```

## Instalação das Dependências

```bash
pip install psycopg2-binary pgvector sentence-transformers
```

Ou use o requirements.txt atualizado:

```bash
pip install -r requirements.txt
```

## Estrutura do Banco de Dados

### Tabela: `documents`

| Coluna       | Tipo      | Descrição                           |
|--------------|-----------|-------------------------------------|
| id           | SERIAL    | ID único do documento               |
| filename     | VARCHAR   | Nome do arquivo                     |
| content      | TEXT      | Conteúdo completo do documento      |
| content_hash | VARCHAR   | Hash SHA-256 (deduplicação)         |
| metadata     | JSONB     | Metadados adicionais                |
| created_at   | TIMESTAMP | Data de criação                     |
| updated_at   | TIMESTAMP | Data de atualização                 |

### Tabela: `document_chunks`

| Coluna       | Tipo       | Descrição                          |
|--------------|------------|------------------------------------|
| id           | SERIAL     | ID único do chunk                  |
| document_id  | INTEGER    | Referência ao documento            |
| chunk_index  | INTEGER    | Índice do chunk no documento       |
| chunk_text   | TEXT       | Texto do chunk                     |
| embedding    | VECTOR(384)| Embedding vetorial (384 dimensões) |
| created_at   | TIMESTAMP  | Data de criação                    |

## Modelo de Embeddings

- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensões**: 384
- **Vantagens**:
  - Rápido e eficiente
  - Boa qualidade para português e inglês
  - Baixo uso de memória (~90MB)

## API Endpoints

### 1. Upload e Indexação Automática de Arquivo

```http
POST /api/upload-file
Content-Type: multipart/form-data

file: <arquivo>
```

**Resposta:**
```json
{
  "filename": "documento.pdf",
  "content_preview": "...",
  "status": "success",
  "message": "Arquivo processado e indexado com sucesso!"
}
```

### 2. Indexar Documento Manualmente

```http
POST /api/rag/index
Content-Type: application/json

{
  "filename": "meu_documento.txt",
  "content": "Conteúdo do documento...",
  "metadata": {
    "author": "João",
    "category": "tecnologia"
  }
}
```

**Resposta:**
```json
{
  "document_id": 1,
  "num_chunks": 5,
  "status": "success",
  "message": "Documento indexado com sucesso!"
}
```

### 3. Buscar Conteúdo Similar

```http
POST /api/rag/search
Content-Type: application/json

{
  "query": "O que é Python?",
  "top_k": 5,
  "min_similarity": 0.6
}
```

**Resposta:**
```json
{
  "results": [
    {
      "id": 1,
      "chunk_text": "Python é uma linguagem...",
      "chunk_index": 0,
      "filename": "python_intro.txt",
      "metadata": {},
      "similarity": 0.85
    }
  ],
  "status": "success"
}
```

### 4. Estatísticas do Banco

```http
GET /api/rag/stats
```

**Resposta:**
```json
{
  "stats": {
    "total_documents": 10,
    "total_chunks": 50,
    "avg_chunks_per_document": 5,
    "embedding_dimensions": 384,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
  },
  "status": "success"
}
```

### 5. Listar Documentos

```http
GET /api/rag/documents?limit=100&offset=0
```

**Resposta:**
```json
{
  "documents": [
    {
      "id": 1,
      "filename": "documento.pdf",
      "content": "...",
      "chunk_count": 5,
      "created_at": "2025-10-14T10:00:00"
    }
  ],
  "status": "success"
}
```

### 6. Deletar Documento

```http
DELETE /api/rag/documents/1
```

**Resposta:**
```json
{
  "status": "success",
  "message": "Documento 1 deletado com sucesso"
}
```

## Integração com Chat

O sistema RAG está **automaticamente integrado** ao endpoint de chat. Quando você faz uma pergunta:

1. O sistema busca os 3 chunks mais relevantes (similaridade > 0.6)
2. Adiciona esse contexto ao prompt do LLM
3. O modelo responde com base no contexto recuperado

**Exemplo de uso:**

1. Upload um documento sobre Python:
   ```bash
   curl -X POST http://localhost:8000/api/upload-file \
     -F "file=@python_tutorial.pdf"
   ```

2. Faça uma pergunta no chat:
   ```json
   {
     "message": "Quais são as principais características do Python?",
     "history": []
   }
   ```

3. O sistema:
   - Busca automaticamente chunks relevantes sobre Python
   - Adiciona ao contexto do LLM
   - Retorna resposta baseada nos documentos indexados

## Testes

Execute o script de teste para verificar a instalação:

```bash
python test_rag.py
```

Este script irá:
- ✓ Verificar conexão com o banco
- ✓ Criar tabelas necessárias
- ✓ Indexar um documento de teste
- ✓ Realizar buscas de similaridade
- ✓ Exibir estatísticas

## Configuração Avançada

### Ajustar Tamanho dos Chunks

No arquivo `rag_db.py`, modifique a função `index_document`:

```python
doc_id, num_chunks = rag_db.index_document(
    filename="doc.txt",
    content=content,
    chunk_size=1000,  # Padrão: 500
    overlap=100       # Padrão: 50
)
```

### Ajustar Parâmetros de Busca

No arquivo `main.py`, na função `chat_with_huggingface_model`:

```python
similar_chunks = rag_db.search_similar_chunks(
    query=request.message,
    top_k=5,              # Padrão: 3
    min_similarity=0.7    # Padrão: 0.6
)
```

### Trocar Modelo de Embeddings

No arquivo `rag_db.py`:

```python
# Modelos alternativos:
# - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (384 dim)
# - 'sentence-transformers/all-mpnet-base-v2' (768 dim, mais lento)
# - 'intfloat/multilingual-e5-small' (384 dim, multilíngue)

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
```

**IMPORTANTE**: Se trocar o modelo, ajuste também a dimensão do vetor:

```sql
CREATE TABLE document_chunks (
    ...
    embedding vector(384),  -- Ajuste para a dimensão do novo modelo
    ...
);
```

## Solução de Problemas

### Erro: "Failed to connect to database"

1. Verifique se o container está rodando:
   ```bash
   docker ps
   ```

2. Verifique os logs:
   ```bash
   docker logs pgvector
   ```

3. Reinicie o container:
   ```bash
   docker restart pgvector
   ```

### Erro: "Extension vector does not exist"

O pgvector não está instalado. Use a imagem Docker correta:
```bash
docker pull ankane/pgvector
```

### Busca não retorna resultados relevantes

1. Verifique a similaridade mínima (tente valores menores)
2. Aumente o `top_k` para retornar mais resultados
3. Verifique se os documentos foram indexados corretamente

## Performance

### Métricas Esperadas

- **Indexação**: ~100-200 chunks/segundo
- **Busca**: <100ms para ~10k chunks
- **Embedding**: ~50ms por query

### Otimizações

1. **Índice IVFFlat**: Já criado automaticamente para acelerar buscas
2. **Batch Indexing**: Indexe múltiplos documentos em lote
3. **Cache de Embeddings**: Considere cachear embeddings frequentes

## Próximos Passos

- [ ] Adicionar suporte para reranking de resultados
- [ ] Implementar cache de embeddings
- [ ] Adicionar suporte para filtros de metadados
- [ ] Criar interface web para gerenciar documentos
- [ ] Adicionar métricas de qualidade das respostas
- [ ] Implementar sistema de feedback do usuário

## Referências

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Sentence Transformers](https://www.sbert.net/)
- [PostgreSQL JSONB](https://www.postgresql.org/docs/current/datatype-json.html)
