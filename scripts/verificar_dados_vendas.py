"""
Script para verificar os dados de vendas inseridos no banco
Mostra algumas estatísticas e exemplos de documentos
"""
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import rag_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("VERIFICAÇÃO DE DADOS DE VENDAS NO BANCO RAG")
    print("=" * 80)
    print()

    # Obter estatísticas gerais
    print("📊 ESTATÍSTICAS DO BANCO DE DADOS")
    print("-" * 80)

    try:
        stats = rag_db.get_database_stats()
        print(f"Total de documentos: {stats['total_documents']}")
        print(f"Total de chunks: {stats['total_chunks']}")
        print(f"Média de chunks por documento: {stats['avg_chunks_per_document']}")
        print(f"Modelo de embeddings: {stats['embedding_model']}")
        print(f"Dimensões dos vetores: {stats['embedding_dimensions']}")
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        return

    # Listar alguns documentos
    print("\n" + "=" * 80)
    print("📄 DOCUMENTOS INDEXADOS (primeiros 10)")
    print("-" * 80)

    try:
        documentos = rag_db.list_documents(limit=10, offset=0)

        for doc in documentos:
            print(f"\nID: {doc['id']}")
            print(f"Arquivo: {doc['filename']}")
            print(f"Chunks: {doc['chunk_count']}")
            print(f"Criado em: {doc['created_at']}")

            # Mostra metadados se existirem
            if doc.get('metadata') and doc['metadata'] != {}:
                print(f"Metadados: {doc['metadata']}")

            # Mostra preview do conteúdo
            preview = doc['content'][:200].replace('\n', ' ')
            print(f"Preview: {preview}...")
            print("-" * 40)
    except Exception as e:
        logger.error(f"Erro ao listar documentos: {e}")

    # Testar busca por similaridade
    print("\n" + "=" * 80)
    print("🔍 TESTE DE BUSCA POR SIMILARIDADE")
    print("-" * 80)

    queries = [
        "vendas de soja",
        "gado nelore",
        "relatório trimestral",
        "Fazenda Santa Rosa",
        "região Mato Grosso"
    ]

    for query in queries:
        print(f"\n🔎 Buscando: '{query}'")
        try:
            resultados = rag_db.search_similar_chunks(
                query=query,
                top_k=3,
                min_similarity=0.3
            )

            if resultados:
                print(f"   Encontrados {len(resultados)} resultados:")
                for i, res in enumerate(resultados[:3], 1):
                    print(f"   {i}. {res['filename']} (similaridade: {res['similarity']:.2f})")
                    # Mostra trecho do chunk
                    chunk_preview = res['chunk_text'][:100].replace('\n', ' ')
                    print(f"      {chunk_preview}...")
            else:
                print("   Nenhum resultado encontrado")
        except Exception as e:
            logger.error(f"Erro ao buscar '{query}': {e}")

    print("\n" + "=" * 80)
    print("✅ VERIFICAÇÃO CONCLUÍDA")
    print("=" * 80)
    print("\n💡 Você pode agora iniciar o servidor e fazer perguntas no chat:")
    print("   python main.py")
    print("\n📌 Exemplos de perguntas:")
    print("   • Quais foram as vendas de soja?")
    print("   • Mostre o relatório do primeiro trimestre")
    print("   • Quais vendas foram feitas para a Fazenda Santa Rosa?")
    print("   • Fale sobre o produto Gado Nelore")
    print()

if __name__ == "__main__":
    main()
