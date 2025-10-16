# Scripts Utilitários

Esta pasta contém scripts auxiliares para gerenciamento do projeto.

## Scripts de Gerenciamento de Modelos

### download_model.py
Download de modelos LLM do Hugging Face.

```bash
python scripts/download_model.py
```

### select_model.py
Seleção interativa do modelo a ser utilizado.

```bash
python scripts/select_model.py
```

## Scripts de Dados de Vendas

### generate_sales_data.py
Gera dados fictícios de vendas agrícolas para testes.

```bash
# Gerar 100 vendas
python scripts/generate_sales_data.py 100
```

### verificar_dados_vendas.py
Verifica os dados de vendas no banco PostgreSQL.

```bash
python scripts/verificar_dados_vendas.py
```

### limpar_dados_vendas.py
Remove todos os dados de vendas do banco.

```bash
python scripts/limpar_dados_vendas.py
```

## Scripts de Migração LlamaIndex

### migrate_to_llamaindex.py
Migra dados do sistema RAG antigo para LlamaIndex.

```bash
# Migração com confirmação
python scripts/migrate_to_llamaindex.py --confirmar
```

### test_llamaindex.py
Testa as funcionalidades do módulo LlamaIndex.

```bash
python scripts/test_llamaindex.py
```

## Estrutura do Projeto

```
ML/
├── main.py                    # Aplicação FastAPI principal
├── rag_llamaindex.py         # Módulo RAG com LlamaIndex
├── model_config.json         # Configuração de modelos
├── requirements.txt          # Dependências Python
├── .env                      # Variáveis de ambiente
├── scripts/                  # Scripts utilitários (esta pasta)
├── instruções/              # Documentação do projeto
├── models/                   # Modelos LLM baixados
└── static/                   # Arquivos estáticos do frontend
```
