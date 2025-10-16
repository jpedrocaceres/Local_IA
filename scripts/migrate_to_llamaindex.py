"""
Script de migração de dados do rag_db.py (antigo) para rag_llamaindex.py (novo)
Migra todos os documentos existentes para o novo sistema LlamaIndex
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
    print("MIGRAÇÃO DE DADOS: rag_db.py → rag_llamaindex.py")
    print("=" * 80)
    print()

    # Confirmação
    print("⚠️  ATENÇÃO: Este script vai migrar todos os documentos do sistema antigo")
    print("   para o novo sistema LlamaIndex.")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--confirmar":
        confirmacao = "sim"
    else:
        try:
            confirmacao = input("Deseja continuar? (digite 'sim' para confirmar): ")
        except EOFError:
            confirmacao = "sim"  # Auto-confirma em modo não-interativo

    if confirmacao.lower() != "sim":
        print("\n❌ Migração cancelada.")
        return 1

    try:
        # Importar módulos
        print("\n📦 Importando módulos...")
        import rag_db  # Sistema antigo
        import rag_llamaindex  # Sistema novo
        print("✅ Módulos importados com sucesso!")

        # Obter estatísticas do sistema antigo
        print("\n📊 Estatísticas do sistema antigo:")
        old_stats = rag_db.get_database_stats()
        print(f"   - Total de documentos: {old_stats.get('total_documents', 0)}")
        print(f"   - Total de chunks: {old_stats.get('total_chunks', 0)}")
        print()

        # Listar documentos para migrar
        print("🔍 Buscando documentos para migrar...")
        todos_documentos = []
        offset = 0
        limit = 100

        while True:
            documentos = rag_db.list_documents(limit=limit, offset=offset)
            if not documentos:
                break
            todos_documentos.extend(documentos)
            offset += limit

        print(f"✅ Encontrados {len(todos_documentos)} documentos para migrar")
        print()

        if len(todos_documentos) == 0:
            print("ℹ️  Nenhum documento para migrar. Encerrando...")
            return 0

        # Migrar cada documento
        print("🔄 Iniciando migração...")
        print("-" * 80)

        migrados = 0
        erros = 0

        for i, doc in enumerate(todos_documentos, 1):
            try:
                # Extrair informações
                filename = doc['filename']
                content = doc['content']
                metadata = doc.get('metadata', {})

                # Adicionar informações extras aos metadados
                if isinstance(metadata, dict):
                    metadata['migrated_from_old_system'] = True
                    metadata['old_doc_id'] = doc['id']
                    metadata['created_at'] = str(doc.get('created_at', ''))

                # Indexar no novo sistema
                doc_id, num_nodes = rag_llamaindex.index_document(
                    filename=filename,
                    content=content,
                    metadata=metadata
                )

                migrados += 1

                if migrados % 10 == 0:
                    print(f"   ✓ {migrados}/{len(todos_documentos)} documentos migrados...")

            except Exception as e:
                erros += 1
                logger.error(f"❌ Erro ao migrar documento {doc.get('id')}: {e}")

        print()
        print("=" * 80)
        print("📊 RESULTADO DA MIGRAÇÃO")
        print("=" * 80)
        print(f"✅ Documentos migrados com sucesso: {migrados}")

        if erros > 0:
            print(f"❌ Erros: {erros}")

        # Estatísticas do novo sistema
        print("\n📊 Estatísticas do novo sistema:")
        new_stats = rag_llamaindex.get_database_stats()
        for key, value in new_stats.items():
            print(f"   - {key}: {value}")

        print()
        print("=" * 80)
        print("✅ MIGRAÇÃO CONCLUÍDA!")
        print("=" * 80)
        print()

        print("💡 Próximos passos:")
        print("   1. Teste o novo sistema: python test_llamaindex.py")
        print("   2. Atualize main.py para usar rag_llamaindex ao invés de rag_db")
        print("   3. Se tudo funcionar, você pode remover o sistema antigo")
        print()

        print("⚠️  IMPORTANTE:")
        print("   - O sistema antigo (rag_db.py) ainda está ativo")
        print("   - Os dados originais NÃO foram deletados")
        print("   - Você pode continuar usando o sistema antigo se necessário")
        print()

        return 0

    except Exception as e:
        logger.error(f"❌ Erro durante a migração: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
