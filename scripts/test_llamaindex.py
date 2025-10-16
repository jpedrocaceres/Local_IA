"""
Script de teste para o módulo RAG com LlamaIndex
"""
import sys
import io
import logging

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("TESTE DO MÓDULO RAG COM LLAMAINDEX")
    print("=" * 80)
    print()

    try:
        # Importar módulo
        print("📦 Importando módulo rag_llamaindex...")
        import rag_llamaindex
        print("✅ Módulo importado com sucesso!")
        print()

        # Testar indexação de documento
        print("📝 Teste 1: Indexar documento de exemplo")
        print("-" * 80)

        test_content = """
        VENDA DE TESTE - LlamaIndex
        ================================================================================

        Data da Venda: 16/10/2025
        Status: Teste

        PRODUTO: Soja Premium
        Quantidade: 500 sacas
        Preço Unitário: R$ 150,00
        Valor Total: R$ 75.000,00

        CLIENTE: Fazenda Teste LlamaIndex
        Região: São Paulo
        Vendedor Responsável: Sistema Teste

        Forma de Pagamento: À vista
        Categorias: grãos, agricultura, teste

        Observações: Esta é uma venda de teste para verificar o funcionamento
        do sistema RAG com LlamaIndex. O produto é soja de alta qualidade.
        """

        doc_id, num_nodes = rag_llamaindex.index_document(
            filename="teste_llamaindex.txt",
            content=test_content,
            metadata={
                "tipo": "teste",
                "data": "16/10/2025",
                "produto": "Soja Premium"
            }
        )

        print(f"✅ Documento indexado!")
        print(f"   - ID: {doc_id}")
        print(f"   - Nodes: {num_nodes}")
        print()

        # Testar busca
        print("🔍 Teste 2: Buscar documentos similares")
        print("-" * 80)

        queries = [
            "vendas de soja",
            "Fazenda Teste",
            "pagamento à vista"
        ]

        for query in queries:
            print(f"\n🔎 Query: '{query}'")
            results = rag_llamaindex.search_similar_chunks(
                query=query,
                top_k=3,
                min_similarity=0.3
            )

            if results:
                print(f"   ✅ Encontrados {len(results)} resultados:")
                for i, res in enumerate(results, 1):
                    print(f"   {i}. {res['filename']} (similaridade: {res['similarity']:.3f})")
                    preview = res['chunk_text'][:100].replace('\n', ' ')
                    print(f"      Preview: {preview}...")
            else:
                print("   ℹ️  Nenhum resultado encontrado")

        print()

        # Testar query engine
        print("🤖 Teste 3: Query Engine (RAG completo)")
        print("-" * 80)

        query_engine = rag_llamaindex.create_query_engine(
            top_k=3,
            similarity_threshold=0.3,
            response_mode="compact"
        )

        query = "Qual foi o valor da venda de soja?"
        print(f"\n❓ Pergunta: {query}")
        print("   (Nota: Resposta será gerada pelo LLM configurado)")

        # O query engine precisa de um LLM configurado
        # Por enquanto, apenas mostramos que foi criado
        print(f"   ✅ Query engine criado com sucesso")
        print(f"   ℹ️  Para usar, configure um LLM em Settings.llm")

        print()

        # Estatísticas
        print("📊 Teste 4: Estatísticas do banco")
        print("-" * 80)

        stats = rag_llamaindex.get_database_stats()
        print("\n📈 Estatísticas:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")

        print()
        print("=" * 80)
        print("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 80)
        print()

        print("💡 Próximos passos:")
        print("   1. Configure um LLM em Settings.llm para usar query engine")
        print("   2. Indexe mais documentos usando index_document()")
        print("   3. Use search_similar_chunks() ou create_query_engine() para buscar")
        print()

    except Exception as e:
        logger.error(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
