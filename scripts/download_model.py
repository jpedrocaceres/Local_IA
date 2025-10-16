#!/usr/bin/env python3
"""
Script para baixar Llama 3.1 8B Instruct do Hugging Face
ATENÇÃO: Requer autenticação no Hugging Face
"""
from huggingface_hub import snapshot_download
import os
import sys

# Configurar encoding para UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Diretório de destino
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# Llama 3.1 8B Instruct - Modelo oficial da Meta
# REQUER AUTENTICAÇÃO: Execute 'huggingface-cli login' antes de usar este script
# E aceite os termos em: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
folder_name = "llama3.1-8b"

print("=" * 80)
print("DOWNLOAD DO LLAMA 3.1 8B INSTRUCT")
print("=" * 80)
print(f"\n📦 Modelo: {model_name}")
print(f"📁 Destino: {os.path.join(models_dir, folder_name)}")
print(f"📊 Tamanho estimado: ~16 GB (FP16)")
print(f"\n⚠️  ATENÇÃO: Este modelo requer autenticação no Hugging Face!")
print(f"\n🔑 Pré-requisitos:")
print(f"   1. Criar conta no Hugging Face: https://huggingface.co")
print(f"   2. Aceitar termos: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
print(f"   3. Login: huggingface-cli login")
print(f"\n⏱️  Este processo pode levar de 10-30 minutos dependendo da sua conexão...\n")

try:
    # Download do modelo
    local_path = snapshot_download(
        repo_id=model_name,
        local_dir=os.path.join(models_dir, folder_name),
        resume_download=True
    )

    print("\n" + "=" * 80)
    print("✅ DOWNLOAD CONCLUÍDO COM SUCESSO!")
    print("=" * 80)
    print(f"\n📁 Localização: {local_path}")
    print(f"\n📊 Especificações do Llama 3.1 8B Instruct:")
    print(f"   • Parâmetros: 8 bilhões")
    print(f"   • Arquitetura: Llama 3.1")
    print(f"   • Contexto: 128K tokens")
    print(f"   • Formato: FP16 (~16 GB)")
    print(f"   • Uso: Chat, instruções, reasoning")
    print(f"\n🚀 Você pode usar este modelo agora!")
    print(f"   Execute: python main.py")

except Exception as e:
    error_msg = str(e)
    print("\n" + "=" * 80)
    print("❌ ERRO AO BAIXAR O MODELO")
    print("=" * 80)
    print(f"\n🔴 Erro: {error_msg}")

    if "401" in error_msg or "gated" in error_msg.lower() or "authentication" in error_msg.lower():
        print("\n🔑 ERRO DE AUTENTICAÇÃO DETECTADO")
        print("\nPara resolver este problema:")
        print("1. Crie uma conta no Hugging Face: https://huggingface.co/join")
        print("2. Aceite os termos de uso do modelo:")
        print("   https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        print("3. Instale a CLI: pip install huggingface_hub")
        print("4. Faça login: huggingface-cli login")
        print("5. Execute este script novamente")
    else:
        print("\n💡 Soluções possíveis:")
        print("1. Verificar sua conexão com a internet")
        print("2. Liberar espaço em disco (necessário ~16-20 GB)")
        print("3. Atualizar dependências: pip install --upgrade huggingface_hub")
        print("4. Usar Ollama como alternativa: ollama pull llama3.1:8b")

    print("\n📖 Consulte a documentação: README.md")
    sys.exit(1)
