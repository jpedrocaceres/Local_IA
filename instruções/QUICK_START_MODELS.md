# Guia Rápido: Seleção de Modelos

## 🚀 Início Rápido em 3 Passos

### 1️⃣ Escolher o Modelo

```bash
python select_model.py
```

### 2️⃣ Configurar o Modelo

#### Opção A: Llama 3.1 (Chat Geral)
```bash
ollama pull llama3.1:8b
```

#### Opção B: TRM (Raciocínio)
```bash
python setup_trm.py
```

### 3️⃣ Iniciar o Servidor

```bash
python main.py
```

Acesse: http://localhost:8000

---

## 🤔 Qual Modelo Escolher?

### Use **Llama 3.1** para:
- ✅ Conversação geral
- ✅ Responder perguntas sobre qualquer assunto
- ✅ Gerar textos, códigos, artigos
- ✅ Seguir instruções complexas
- ✅ Contexto e nuances de linguagem

**Requer:** 8-16GB VRAM

### Use **TRM** para:
- ✅ Puzzles lógicos (Sudoku, Mazes)
- ✅ Reconhecimento de padrões (ARC-AGI)
- ✅ Problemas de raciocínio
- ✅ Tarefas que precisam de refinamento iterativo
- ✅ Baixo uso de memória GPU

**Requer:** 2-4GB VRAM

---

## 📊 Comparação Rápida

| Característica | Llama 3.1 | TRM |
|----------------|-----------|-----|
| Tamanho | 16GB | 28MB |
| Setup | Fácil | Médio |
| Chat Geral | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Raciocínio | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| VRAM | 8-16GB | 2-4GB |

---

## 🔄 Trocar de Modelo

1. Parar servidor: `Ctrl+C`
2. Selecionar: `python select_model.py`
3. Reiniciar: `python main.py`

---

## 📚 Mais Informações

- **Llama 3.1**: [README.md](README.md)
- **TRM**: [TRM_GUIDE.md](TRM_GUIDE.md)
- **Problemas**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
