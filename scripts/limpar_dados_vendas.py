"""
Script para limpar dados de vendas do banco de dados
CUIDADO: Este script remove TODOS os documentos de vendas!
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
    print("⚠️  LIMPEZA DE DADOS DE VENDAS DO BANCO RAG")
    print("=" * 80)
    print()
    print("ATENÇÃO: Este script irá REMOVER TODOS os documentos do banco!")
    print()

    # Mostra estatísticas atuais
    try:
        stats = rag_db.get_database_stats()
        print(f"📊 Estatísticas atuais:")
        print(f"   Total de documentos: {stats['total_documents']}")
        print(f"   Total de chunks: {stats['total_chunks']}")
        print()
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        return

    # Confirmação
    if len(sys.argv) > 1 and sys.argv[1] == "--confirmar":
        confirmacao = "sim"
    else:
        try:
            confirmacao = input("Tem certeza que deseja remover TODOS os documentos? (digite 'sim' para confirmar): ")
        except EOFError:
            print("\nOperação cancelada.")
            return

    if confirmacao.lower() != "sim":
        print("\n❌ Operação cancelada.")
        return

    # Lista todos os documentos
    print("\n🔍 Buscando documentos...")
    try:
        # Busca todos os documentos (em lotes se necessário)
        todos_documentos = []
        offset = 0
        limit = 100

        while True:
            documentos = rag_db.list_documents(limit=limit, offset=offset)
            if not documentos:
                break
            todos_documentos.extend(documentos)
            offset += limit

        print(f"📄 Encontrados {len(todos_documentos)} documentos para remover")
        print()

        # Remove cada documento
        removidos = 0
        erros = 0

        for doc in todos_documentos:
            try:
                sucesso = rag_db.delete_document(doc['id'])
                if sucesso:
                    removidos += 1
                    if removidos % 10 == 0:
                        print(f"   ✓ {removidos}/{len(todos_documentos)} documentos removidos...")
                else:
                    erros += 1
                    logger.warning(f"Documento {doc['id']} não foi encontrado")
            except Exception as e:
                erros += 1
                logger.error(f"Erro ao remover documento {doc['id']}: {e}")

        print()
        print("=" * 80)
        print("📊 RESULTADO DA LIMPEZA")
        print("=" * 80)
        print(f"✅ Documentos removidos: {removidos}")
        if erros > 0:
            print(f"❌ Erros: {erros}")

        # Mostra estatísticas finais
        try:
            stats_final = rag_db.get_database_stats()
            print()
            print(f"📊 Estatísticas após limpeza:")
            print(f"   Total de documentos: {stats_final['total_documents']}")
            print(f"   Total de chunks: {stats_final['total_chunks']}")
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas finais: {e}")

        print()
        print("=" * 80)
        print("✅ LIMPEZA CONCLUÍDA")
        print("=" * 80)
        print()
        print("💡 Para gerar novos dados de teste, execute:")
        print("   python generate_sales_data.py 100")
        print()

    except Exception as e:
        logger.error(f"Erro durante a limpeza: {e}")
        return

if __name__ == "__main__":
    main()
