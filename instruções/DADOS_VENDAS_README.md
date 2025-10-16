# Dados de Vendas para Testes - Sistema RAG

## Resumo

Este documento descreve o script de geração de dados de vendas fictícios para testes do sistema RAG (Retrieval-Augmented Generation) com PostgreSQL + pgvector.

## Scripts Disponíveis

### 1. `generate_sales_data.py`

Script principal para gerar dados de vendas fictícios e indexá-los no banco de dados vetorial.

**Uso:**
```bash
# Gerar número padrão de vendas (50)
python generate_sales_data.py

# Gerar quantidade específica de vendas
python generate_sales_data.py 100
```

**O que o script gera:**

1. **Vendas individuais** - Documentos detalhados de cada venda contendo:
   - Data da venda
   - Produto vendido
   - Quantidade e preço
   - Cliente e região
   - Vendedor responsável
   - Forma de pagamento
   - Status da venda

2. **Relatórios consolidados**:
   - Relatório trimestral
   - Relatório anual
   - Análises por produto
   - Análises por região

3. **Análises de produtos** - Documentos com informações detalhadas sobre:
   - Soja
   - Milho
   - Café Arábica
   - Café Robusta
   - Gado Nelore
   - Gado Angus
   - Fertilizantes
   - Defensivos agrícolas
   - Maquinário agrícola

### 2. `verificar_dados_vendas.py`

Script para verificar os dados inseridos no banco e testar buscas por similaridade.

**Uso:**
```bash
python verificar_dados_vendas.py
```

**O que o script faz:**
- Mostra estatísticas do banco de dados
- Lista os primeiros documentos indexados
- Realiza testes de busca por similaridade
- Fornece exemplos de perguntas para testar o RAG

## Dados Gerados (Última Execução)

### Estatísticas
- **Total de documentos:** 109
- **Total de chunks:** 229
- **Média de chunks por documento:** 2
- **Número de vendas:** 100
- **Valor total:** R$ 50.613.778,67
- **Ticket médio:** R$ 506.137,79

### Produtos Disponíveis

| Produto | Unidade | Faixa de Preço | Categorias |
|---------|---------|----------------|------------|
| Soja | sacas | R$ 120-180 | grãos, agricultura, commodities |
| Milho | sacas | R$ 80-120 | grãos, agricultura, commodities |
| Café Arábica | sacas | R$ 800-1.200 | café, agricultura, commodities |
| Café Robusta | sacas | R$ 600-900 | café, agricultura, commodities |
| Gado Nelore | cabeças | R$ 2.500-4.000 | pecuária, gado, carne |
| Gado Angus | cabeças | R$ 3.000-5.000 | pecuária, gado, carne premium |
| Fertilizante NPK | toneladas | R$ 2.000-3.500 | insumos, fertilizantes |
| Defensivo Agrícola | litros | R$ 50-150 | insumos, defensivos |
| Trator John Deere | unidades | R$ 250.000-500.000 | maquinário, equipamentos |
| Colheitadeira | unidades | R$ 800.000-1.500.000 | maquinário, equipamentos |

### Clientes Cadastrados
- Fazenda Santa Rosa
- Agropecuária Pantanal Ltda
- Cooperativa Agrícola do Cerrado
- Fazenda Boa Vista
- Rancho Alegre Agropecuária
- Fazenda São Francisco
- Agronegócio Mato Grosso S.A.
- Fazenda Esperança
- Cooperativa dos Produtores de Grãos
- Fazenda Três Irmãos
- AgroIndustrial Campo Grande
- Fazenda Progresso
- Pecuária Forte Ltda
- Fazenda Nova Era
- Cooperativa Agrícola Regional

### Regiões Cobertas
- Mato Grosso do Sul
- Mato Grosso
- Goiás
- São Paulo
- Paraná
- Bahia
- Tocantins
- Minas Gerais

## Estrutura dos Documentos

### Formato de Venda Individual
```
VENDA #0001
================================================================================

Data da Venda: 15/03/2025
Status: Concluída

PRODUTO: Soja
Quantidade: 500 sacas
Preço Unitário: R$ 150,00
Valor Total: R$ 75.000,00

CLIENTE: Fazenda Santa Rosa
Região: Mato Grosso
Vendedor Responsável: João Silva

Forma de Pagamento: 30 dias
Categorias: grãos, agricultura, commodities

Observações: Venda de Soja realizada na região Mato Grosso
pelo vendedor João Silva. Cliente Fazenda Santa Rosa adquiriu
500 sacas com pagamento 30 dias.
```

## Testando o Sistema RAG

### Iniciar o Servidor
```bash
python main.py
```

### Exemplos de Perguntas

#### Consultas sobre vendas específicas:
- "Quais foram as vendas de soja?"
- "Mostre vendas de gado nelore"
- "Quais vendas foram feitas para a Fazenda Santa Rosa?"
- "Liste vendas na região de Mato Grosso"

#### Análises e relatórios:
- "Mostre o relatório do primeiro trimestre"
- "Qual o valor total de vendas por região?"
- "Qual vendedor teve mais vendas?"
- "Quais são os principais produtos vendidos?"

#### Informações sobre produtos:
- "Fale sobre o produto Gado Nelore"
- "Quais são as características da soja?"
- "Qual a faixa de preço do café arábica?"
- "Mostre análise de produtos agrícolas"

#### Consultas complexas:
- "Qual foi o ticket médio das vendas?"
- "Quais clientes compraram maquinário?"
- "Liste vendas com pagamento à vista"
- "Mostre vendas canceladas"

## Configuração do Banco de Dados

O sistema utiliza PostgreSQL com a extensão pgvector para armazenamento e busca vetorial.

### Configuração (.env)
```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=vetorial_bd
DB_USER=postgres
DB_PASSWORD=sua_senha

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RAG_TOP_K=3
RAG_MIN_SIMILARITY=0.6
```

## Tecnologias Utilizadas

- **PostgreSQL** - Banco de dados relacional
- **pgvector** - Extensão para busca vetorial
- **sentence-transformers** - Geração de embeddings (all-MiniLM-L6-v2)
- **Python** - Linguagem de programação
- **psycopg2** - Driver PostgreSQL para Python

## Próximos Passos

1. Iniciar o servidor: `python main.py`
2. Acessar a interface web
3. Fazer upload de documentos adicionais (se necessário)
4. Testar perguntas usando o sistema RAG
5. Avaliar a qualidade das respostas
6. Ajustar parâmetros de similaridade conforme necessário

## Notas

- Os dados gerados são **fictícios** e destinados apenas para **testes**
- As datas das vendas são geradas aleatoriamente nos últimos 365 dias
- Os preços variam dentro de faixas realistas para cada produto
- O sistema usa **busca vetorial por similaridade semântica**
- Cada documento é dividido em chunks para melhor indexação
- Os embeddings têm **384 dimensões** (modelo all-MiniLM-L6-v2)

## Suporte

Para problemas ou dúvidas:
1. Verifique se o PostgreSQL está rodando
2. Confirme que a extensão pgvector está instalada
3. Verifique as configurações no arquivo `.env`
4. Execute `verificar_dados_vendas.py` para diagnóstico
