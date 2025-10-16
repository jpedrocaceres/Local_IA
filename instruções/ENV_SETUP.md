# Configuração de Variáveis de Ambiente

Este projeto usa variáveis de ambiente para configuração do PostgreSQL/pgvector e do sistema RAG.

## Setup Inicial

### 1. Criar arquivo .env

Copie o arquivo `.env.example` para `.env`:

```bash
cp .env.example .env
```

Ou no Windows:
```cmd
copy .env.example .env
```

### 2. Configurar PostgreSQL com Docker

Execute o comando Docker para iniciar o PostgreSQL com pgvector:

```bash
docker run -d \
  --name pgvector \
  -p 5433:5432 \
  -e POSTGRES_PASSWORD=agro123 \
  -e POSTGRES_DB=vetorial_bd \
  -v pgvector_data:/var/lib/postgresql/data \
  ankane/pgvector
```

### 3. Atualizar .env com suas configurações

Edite o arquivo `.env` com suas credenciais:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5433
DB_NAME=vetorial_bd
DB_USER=postgres
DB_PASSWORD=agro123
```

## Variáveis de Ambiente Disponíveis

### Configuração do Banco de Dados

| Variável | Descrição | Padrão | Obrigatória |
|----------|-----------|--------|-------------|
| `DB_HOST` | Hostname do PostgreSQL | `localhost` | Sim |
| `DB_PORT` | Porta do PostgreSQL | `5433` | Sim |
| `DB_NAME` | Nome do banco de dados | `vetorial_bd` | Sim |
| `DB_USER` | Usuário do banco | `postgres` | Sim |
| `DB_PASSWORD` | Senha do banco | `agro123` | Sim |

### Configuração do RAG

| Variável | Descrição | Padrão | Tipo |
|----------|-----------|--------|------|
| `EMBEDDING_MODEL` | Modelo de embeddings | `sentence-transformers/all-MiniLM-L6-v2` | String |
| `CHUNK_SIZE` | Tamanho dos chunks de texto | `500` | Integer |
| `CHUNK_OVERLAP` | Sobreposição entre chunks | `50` | Integer |
| `RAG_TOP_K` | Número de chunks retornados na busca | `3` | Integer |
| `RAG_MIN_SIMILARITY` | Similaridade mínima para busca (0-1) | `0.6` | Float |

### Configuração de Geração do LLM

| Variável | Descrição | Padrão | Tipo |
|----------|-----------|--------|------|
| `MAX_NEW_TOKENS` | Número máximo de tokens na resposta | `1024` | Integer |
| `TEMPERATURE` | Controla aleatoriedade (0.0-2.0) | `0.7` | Float |
| `TOP_P` | Nucleus sampling (0.0-1.0) | `0.9` | Float |

**Sobre MAX_NEW_TOKENS:**
- `256` - Respostas curtas (~200 palavras) ⚠️ **Pode cortar respostas**
- `512` - Respostas médias (~400 palavras)
- `1024` - Respostas longas (~800 palavras) ✅ **Recomendado**
- `2048` - Respostas muito longas (~1600 palavras)

**Sobre TEMPERATURE:**
- `0.0-0.3` - Respostas mais determinísticas e focadas
- `0.7` - Balanceado (padrão)
- `1.0-2.0` - Respostas mais criativas e variadas

**Sobre TOP_P:**
- `0.5` - Mais conservador
- `0.9` - Balanceado (padrão)
- `0.95-1.0` - Mais diverso

### Configuração Avançada (Opcional)

| Variável | Descrição | Padrão | Tipo |
|----------|-----------|--------|------|
| `DB_POOL_MIN_SIZE` | Tamanho mínimo do pool de conexões | `1` | Integer |
| `DB_POOL_MAX_SIZE` | Tamanho máximo do pool de conexões | `10` | Integer |

## Como as Variáveis São Usadas

### rag_db.py

O módulo `rag_db.py` carrega automaticamente as variáveis de ambiente:

```python
from dotenv import load_dotenv
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'vetorial_bd'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'agro123')
}
```

### main.py

O `main.py` também carrega as variáveis para configurar o RAG no chat:

```python
from dotenv import load_dotenv
load_dotenv()

# No chat
rag_top_k = int(os.getenv('RAG_TOP_K', 3))
rag_min_similarity = float(os.getenv('RAG_MIN_SIMILARITY', 0.6))
```

## Exemplos de Configuração

### Desenvolvimento Local

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=vetorial_bd
DB_USER=postgres
DB_PASSWORD=agro123

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RAG_TOP_K=3
RAG_MIN_SIMILARITY=0.6
```

### Produção (servidor remoto)

```env
DB_HOST=db.example.com
DB_PORT=5432
DB_NAME=production_db
DB_USER=app_user
DB_PASSWORD=secure_password_here

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RAG_TOP_K=5
RAG_MIN_SIMILARITY=0.7
```

### Chunks Maiores para Documentos Técnicos

```env
# ... outras configurações ...

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
RAG_TOP_K=5
RAG_MIN_SIMILARITY=0.5
```

### Busca Mais Restritiva

```env
# ... outras configurações ...

RAG_TOP_K=2
RAG_MIN_SIMILARITY=0.8
```

## Verificação da Configuração

### 1. Verificar conexão com o banco

```bash
docker exec -it pgvector psql -U postgres -d vetorial_bd
```

Dentro do psql:
```sql
-- Verificar extensão pgvector
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Listar tabelas
\dt

-- Ver documentos indexados
SELECT id, filename, created_at FROM documents;
```

### 2. Testar o sistema RAG

```bash
python test_rag.py
```

Você deve ver:
```
✓ Database initialized successfully
✓ Document indexed successfully!
✓ Statistics retrieved successfully
```

### 3. Verificar logs ao iniciar o servidor

```bash
python main.py
```

Procure nos logs:
```
INFO:rag_db:RAG Database Configuration:
INFO:rag_db:  - Database: postgres@localhost:5433/vetorial_bd
INFO:rag_db:  - Embedding Model: sentence-transformers/all-MiniLM-L6-v2
INFO:rag_db:  - Chunk Size: 500 chars
INFO:rag_db:  - Chunk Overlap: 50 chars
```

## Modelos de Embeddings Alternativos

Você pode trocar o modelo de embeddings editando a variável `EMBEDDING_MODEL`:

### Multilíngue (Português + Inglês)

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Maior Qualidade (mais lento)

```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```
**Nota:** Este modelo usa 768 dimensões. Você precisará recriar a tabela:

```sql
DROP TABLE document_chunks;
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(768),  -- 768 dimensões para all-mpnet-base-v2
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Especializado em Português

```env
EMBEDDING_MODEL=neuralmind/bert-base-portuguese-cased
```

## Troubleshooting

### Erro: "Failed to connect to database"

1. Verifique se o container está rodando:
   ```bash
   docker ps | grep pgvector
   ```

2. Verifique as variáveis de ambiente:
   ```bash
   cat .env
   ```

3. Teste a conexão:
   ```bash
   docker exec -it pgvector psql -U postgres -d vetorial_bd
   ```

### Erro: "Environment variable not found"

Certifique-se de que:
1. O arquivo `.env` existe no diretório raiz do projeto
2. O `python-dotenv` está instalado: `pip install python-dotenv`
3. O código está carregando o `.env`: `load_dotenv()`

### Logs de Debug

Para ver mais detalhes sobre o carregamento das variáveis:

```python
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)  # Mostra arquivos .env carregados

print(f"DB_HOST: {os.getenv('DB_HOST')}")
print(f"DB_PORT: {os.getenv('DB_PORT')}")
```

## Segurança

⚠️ **IMPORTANTE:**

1. **Nunca** commite o arquivo `.env` no Git
2. O `.env` já está no `.gitignore`
3. Use senhas fortes em produção
4. Considere usar secrets managers em produção (AWS Secrets Manager, HashiCorp Vault, etc.)

## Resumo dos Arquivos

- `.env` - **NÃO COMMITAR** - Suas configurações locais
- `.env.example` - Template para criar o `.env`
- `.gitignore` - Já inclui `.env` e `.env.local`
- `rag_db.py` - Carrega configurações do banco e RAG
- `main.py` - Carrega configurações para o chat

## Próximos Passos

Após configurar o `.env`:

1. ✅ Inicie o PostgreSQL: `docker run ...`
2. ✅ Verifique o `.env` com suas credenciais
3. ✅ Teste o sistema: `python test_rag.py`
4. ✅ Inicie o servidor: `python main.py`
5. ✅ Faça upload de documentos e teste o RAG!
