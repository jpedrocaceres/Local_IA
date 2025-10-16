"""
Script para gerar dados de vendas fictícios e indexá-los no RAG
Útil para testar o sistema de busca vetorial com dados realistas
"""
import sys
import io
import random
from datetime import datetime, timedelta

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import rag_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dados fictícios para geração
PRODUTOS = {
    "Soja": {
        "unidade": "sacas",
        "preco_min": 120,
        "preco_max": 180,
        "quantidade_min": 100,
        "quantidade_max": 1000,
        "categorias": ["grãos", "agricultura", "commodities"]
    },
    "Milho": {
        "unidade": "sacas",
        "preco_min": 80,
        "preco_max": 120,
        "quantidade_min": 150,
        "quantidade_max": 1200,
        "categorias": ["grãos", "agricultura", "commodities"]
    },
    "Café Arábica": {
        "unidade": "sacas",
        "preco_min": 800,
        "preco_max": 1200,
        "quantidade_min": 50,
        "quantidade_max": 500,
        "categorias": ["café", "agricultura", "commodities"]
    },
    "Café Robusta": {
        "unidade": "sacas",
        "preco_min": 600,
        "preco_max": 900,
        "quantidade_min": 60,
        "quantidade_max": 600,
        "categorias": ["café", "agricultura", "commodities"]
    },
    "Gado Nelore": {
        "unidade": "cabeças",
        "preco_min": 2500,
        "preco_max": 4000,
        "quantidade_min": 10,
        "quantidade_max": 100,
        "categorias": ["pecuária", "gado", "carne"]
    },
    "Gado Angus": {
        "unidade": "cabeças",
        "preco_min": 3000,
        "preco_max": 5000,
        "quantidade_min": 5,
        "quantidade_max": 80,
        "categorias": ["pecuária", "gado", "carne premium"]
    },
    "Fertilizante NPK": {
        "unidade": "toneladas",
        "preco_min": 2000,
        "preco_max": 3500,
        "quantidade_min": 5,
        "quantidade_max": 50,
        "categorias": ["insumos", "fertilizantes", "agricultura"]
    },
    "Defensivo Agrícola": {
        "unidade": "litros",
        "preco_min": 50,
        "preco_max": 150,
        "quantidade_min": 100,
        "quantidade_max": 1000,
        "categorias": ["insumos", "defensivos", "agricultura"]
    },
    "Trator John Deere": {
        "unidade": "unidades",
        "preco_min": 250000,
        "preco_max": 500000,
        "quantidade_min": 1,
        "quantidade_max": 5,
        "categorias": ["maquinário", "equipamentos", "agricultura"]
    },
    "Colheitadeira": {
        "unidade": "unidades",
        "preco_min": 800000,
        "preco_max": 1500000,
        "quantidade_min": 1,
        "quantidade_max": 3,
        "categorias": ["maquinário", "equipamentos", "agricultura"]
    }
}

CLIENTES = [
    "Fazenda Santa Rosa",
    "Agropecuária Pantanal Ltda",
    "Cooperativa Agrícola do Cerrado",
    "Fazenda Boa Vista",
    "Rancho Alegre Agropecuária",
    "Fazenda São Francisco",
    "Agronegócio Mato Grosso S.A.",
    "Fazenda Esperança",
    "Cooperativa dos Produtores de Grãos",
    "Fazenda Três Irmãos",
    "AgroIndustrial Campo Grande",
    "Fazenda Progresso",
    "Pecuária Forte Ltda",
    "Fazenda Nova Era",
    "Cooperativa Agrícola Regional"
]

VENDEDORES = [
    "João Silva",
    "Maria Santos",
    "Pedro Oliveira",
    "Ana Costa",
    "Carlos Ferreira",
    "Juliana Almeida",
    "Ricardo Souza",
    "Fernanda Lima"
]

REGIOES = [
    "Mato Grosso do Sul",
    "Mato Grosso",
    "Goiás",
    "São Paulo",
    "Paraná",
    "Bahia",
    "Tocantins",
    "Minas Gerais"
]

STATUS_VENDA = ["Concluída", "Em andamento", "Pendente", "Cancelada"]
FORMA_PAGAMENTO = ["À vista", "30 dias", "60 dias", "90 dias", "Parcelado"]

def gerar_data_aleatoria(dias_atras_min=0, dias_atras_max=365):
    """Gera uma data aleatória dentro de um intervalo"""
    dias_atras = random.randint(dias_atras_min, dias_atras_max)
    return datetime.now() - timedelta(days=dias_atras)

def gerar_venda():
    """Gera dados de uma venda fictícia"""
    produto = random.choice(list(PRODUTOS.keys()))
    config = PRODUTOS[produto]

    quantidade = random.randint(config["quantidade_min"], config["quantidade_max"])
    preco_unitario = round(random.uniform(config["preco_min"], config["preco_max"]), 2)
    total = round(quantidade * preco_unitario, 2)

    data_venda = gerar_data_aleatoria(0, 365)

    venda = {
        "data": data_venda.strftime("%d/%m/%Y"),
        "produto": produto,
        "quantidade": quantidade,
        "unidade": config["unidade"],
        "preco_unitario": preco_unitario,
        "total": total,
        "cliente": random.choice(CLIENTES),
        "vendedor": random.choice(VENDEDORES),
        "regiao": random.choice(REGIOES),
        "status": random.choice(STATUS_VENDA),
        "forma_pagamento": random.choice(FORMA_PAGAMENTO),
        "categorias": config["categorias"]
    }

    return venda

def formatar_venda_texto(venda, numero):
    """Formata os dados da venda em texto descritivo"""
    texto = f"""
VENDA #{numero:04d}
================================================================================

Data da Venda: {venda['data']}
Status: {venda['status']}

PRODUTO: {venda['produto']}
Quantidade: {venda['quantidade']} {venda['unidade']}
Preço Unitário: R$ {venda['preco_unitario']:,.2f}
Valor Total: R$ {venda['total']:,.2f}

CLIENTE: {venda['cliente']}
Região: {venda['regiao']}
Vendedor Responsável: {venda['vendedor']}

Forma de Pagamento: {venda['forma_pagamento']}
Categorias: {', '.join(venda['categorias'])}

Observações: Venda de {venda['produto']} realizada na região {venda['regiao']}
pelo vendedor {venda['vendedor']}. Cliente {venda['cliente']} adquiriu
{venda['quantidade']} {venda['unidade']} com pagamento {venda['forma_pagamento']}.
"""
    return texto

def gerar_relatorio_periodo(vendas, periodo):
    """Gera um relatório consolidado de um período"""
    total_vendas = len(vendas)
    valor_total = sum(v['total'] for v in vendas)

    # Agrupa por produto
    por_produto = {}
    for v in vendas:
        produto = v['produto']
        if produto not in por_produto:
            por_produto[produto] = {'qtd': 0, 'valor': 0}
        por_produto[produto]['qtd'] += v['quantidade']
        por_produto[produto]['valor'] += v['total']

    # Agrupa por região
    por_regiao = {}
    for v in vendas:
        regiao = v['regiao']
        if regiao not in por_regiao:
            por_regiao[regiao] = {'qtd': 0, 'valor': 0}
        por_regiao[regiao]['qtd'] += 1
        por_regiao[regiao]['valor'] += v['total']

    texto = f"""
RELATÓRIO DE VENDAS - {periodo}
================================================================================

RESUMO EXECUTIVO
----------------
Total de Vendas: {total_vendas}
Valor Total: R$ {valor_total:,.2f}
Ticket Médio: R$ {valor_total/total_vendas:,.2f}

VENDAS POR PRODUTO
------------------
"""

    for produto, dados in sorted(por_produto.items(), key=lambda x: x[1]['valor'], reverse=True):
        config = PRODUTOS[produto]
        texto += f"\n{produto}:\n"
        texto += f"  - Quantidade: {dados['qtd']} {config['unidade']}\n"
        texto += f"  - Valor Total: R$ {dados['valor']:,.2f}\n"
        texto += f"  - Participação: {(dados['valor']/valor_total)*100:.1f}%\n"

    texto += "\n\nVENDAS POR REGIÃO\n------------------\n"

    for regiao, dados in sorted(por_regiao.items(), key=lambda x: x[1]['valor'], reverse=True):
        texto += f"\n{regiao}:\n"
        texto += f"  - Número de Vendas: {dados['qtd']}\n"
        texto += f"  - Valor Total: R$ {dados['valor']:,.2f}\n"
        texto += f"  - Participação: {(dados['valor']/valor_total)*100:.1f}%\n"

    return texto

def gerar_analise_produto(produto_nome):
    """Gera uma análise detalhada de um produto"""
    config = PRODUTOS[produto_nome]

    texto = f"""
ANÁLISE DE PRODUTO: {produto_nome}
================================================================================

INFORMAÇÕES GERAIS
------------------
Categoria: {', '.join(config['categorias'])}
Unidade de Medida: {config['unidade']}

FAIXA DE PREÇO
--------------
Preço Mínimo: R$ {config['preco_min']:,.2f}
Preço Máximo: R$ {config['preco_max']:,.2f}
Preço Médio: R$ {(config['preco_min'] + config['preco_max'])/2:,.2f}

VOLUME DE VENDAS
----------------
Quantidade Mínima: {config['quantidade_min']} {config['unidade']}
Quantidade Máxima: {config['quantidade_max']} {config['unidade']}

APLICAÇÕES E MERCADO
--------------------
"""

    # Adiciona informações específicas por tipo de produto
    if "grãos" in config['categorias']:
        texto += f"""
{produto_nome} é uma commodity agrícola amplamente comercializada no mercado
brasileiro e internacional. Principal produto de exportação, é cultivado
principalmente nas regiões Centro-Oeste e Sul do Brasil. O preço varia conforme
a safra, clima e demanda internacional.
"""
    elif "gado" in config['categorias']:
        texto += f"""
{produto_nome} é uma raça de gado de corte muito valorizada no mercado brasileiro.
A pecuária representa uma importante atividade econômica, especialmente nas regiões
do Pantanal e Centro-Oeste. O valor dos animais varia conforme idade, peso e
qualidade genética.
"""
    elif "insumos" in config['categorias']:
        texto += f"""
{produto_nome} é um insumo essencial para a agricultura moderna. Sua utilização
correta aumenta significativamente a produtividade das lavouras. A demanda é
sazonal, concentrando-se nos períodos de plantio e desenvolvimento das culturas.
"""
    elif "maquinário" in config['categorias']:
        texto += f"""
{produto_nome} representa um investimento significativo para produtores rurais.
Equipamentos modernos aumentam a eficiência operacional e reduzem custos de
produção no longo prazo. A decisão de compra geralmente envolve análise de
financiamento e retorno sobre investimento.
"""

    return texto

def main():
    """Função principal"""
    print("=" * 80)
    print("GERADOR DE DADOS DE VENDAS FICTÍCIOS PARA RAG")
    print("=" * 80)
    print()

    # Pergunta quantas vendas gerar ou usa argumento da linha de comando
    num_vendas = 50  # padrão

    if len(sys.argv) > 1:
        try:
            num_vendas = int(sys.argv[1])
        except ValueError:
            print(f"Argumento inválido '{sys.argv[1]}', usando padrão: 50")
            num_vendas = 50
    else:
        try:
            resposta = input("Quantas vendas você deseja gerar? (padrão: 50): ")
            if resposta.strip():
                num_vendas = int(resposta)
        except (ValueError, EOFError):
            num_vendas = 50

    print(f"\n📊 Gerando {num_vendas} vendas fictícias...")

    # Gera vendas individuais
    vendas = []
    documentos_gerados = 0

    for i in range(num_vendas):
        venda = gerar_venda()
        vendas.append(venda)

        # Formata e indexa a venda
        texto_venda = formatar_venda_texto(venda, i + 1)

        try:
            doc_id, num_chunks = rag_db.index_document(
                filename=f"venda_{i+1:04d}.txt",
                content=texto_venda,
                metadata={
                    "tipo": "venda",
                    "numero": i + 1,
                    "produto": venda['produto'],
                    "cliente": venda['cliente'],
                    "regiao": venda['regiao'],
                    "data": venda['data'],
                    "valor": venda['total']
                }
            )
            documentos_gerados += 1

            if (i + 1) % 10 == 0:
                print(f"  ✓ {i + 1}/{num_vendas} vendas indexadas...")

        except Exception as e:
            logger.error(f"Erro ao indexar venda {i+1}: {e}")

    print(f"\n✅ {documentos_gerados} vendas indexadas com sucesso!")

    # Gera relatórios consolidados
    print("\n📈 Gerando relatórios consolidados...")

    # Relatório trimestral
    vendas_trimestre = vendas[:len(vendas)//4] if len(vendas) >= 4 else vendas
    relatorio_trimestre = gerar_relatorio_periodo(vendas_trimestre, "1º TRIMESTRE 2025")

    try:
        doc_id, num_chunks = rag_db.index_document(
            filename="relatorio_trimestral_2025_Q1.txt",
            content=relatorio_trimestre,
            metadata={
                "tipo": "relatorio",
                "periodo": "trimestral",
                "ano": 2025,
                "trimestre": 1
            }
        )
        print("  ✓ Relatório trimestral indexado")
        documentos_gerados += 1
    except Exception as e:
        logger.error(f"Erro ao indexar relatório trimestral: {e}")

    # Relatório anual
    relatorio_anual = gerar_relatorio_periodo(vendas, "ANO 2024")

    try:
        doc_id, num_chunks = rag_db.index_document(
            filename="relatorio_anual_2024.txt",
            content=relatorio_anual,
            metadata={
                "tipo": "relatorio",
                "periodo": "anual",
                "ano": 2024
            }
        )
        print("  ✓ Relatório anual indexado")
        documentos_gerados += 1
    except Exception as e:
        logger.error(f"Erro ao indexar relatório anual: {e}")

    # Gera análises de produtos
    print("\n📦 Gerando análises de produtos...")

    for produto in list(PRODUTOS.keys())[:5]:  # Primeiros 5 produtos
        analise = gerar_analise_produto(produto)

        try:
            doc_id, num_chunks = rag_db.index_document(
                filename=f"analise_produto_{produto.lower().replace(' ', '_')}.txt",
                content=analise,
                metadata={
                    "tipo": "analise",
                    "categoria": "produto",
                    "produto": produto
                }
            )
            print(f"  ✓ Análise de {produto} indexada")
            documentos_gerados += 1
        except Exception as e:
            logger.error(f"Erro ao indexar análise de {produto}: {e}")

    # Estatísticas finais
    print("\n" + "=" * 80)
    print("ESTATÍSTICAS FINAIS")
    print("=" * 80)

    try:
        stats = rag_db.get_database_stats()
        print(f"\n📊 Banco de Dados RAG:")
        print(f"  - Total de documentos: {stats['total_documents']}")
        print(f"  - Total de chunks: {stats['total_chunks']}")
        print(f"  - Média de chunks por documento: {stats['avg_chunks_per_document']}")
        print(f"  - Modelo de embeddings: {stats['embedding_model']}")
        print(f"  - Dimensões do vetor: {stats['embedding_dimensions']}")
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")

    print(f"\n💰 Dados de Vendas Gerados:")
    print(f"  - Número de vendas: {len(vendas)}")
    print(f"  - Valor total: R$ {sum(v['total'] for v in vendas):,.2f}")
    print(f"  - Ticket médio: R$ {sum(v['total'] for v in vendas)/len(vendas):,.2f}")

    print("\n" + "=" * 80)
    print("🎉 DADOS GERADOS E INDEXADOS COM SUCESSO!")
    print("=" * 80)

    print("\n📌 Exemplos de perguntas para testar o RAG:")
    print("  • Quais foram as vendas de soja?")
    print("  • Mostre o relatório do primeiro trimestre")
    print("  • Qual o valor total de vendas por região?")
    print("  • Quais vendas foram feitas para a Fazenda Santa Rosa?")
    print("  • Qual vendedor teve mais vendas?")
    print("  • Fale sobre o produto Gado Nelore")
    print("  • Quais são os principais produtos vendidos?")
    print("  • Mostre análise de produtos agrícolas")

    print("\n💡 Inicie o servidor e faça perguntas no chat:")
    print("   python main.py")
    print()

if __name__ == "__main__":
    main()
