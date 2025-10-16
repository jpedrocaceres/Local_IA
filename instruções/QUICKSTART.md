# 🚀 Início Rápido - Sistema LLaMA Chat

## ✅ Status do Projeto

**Modelo Instalado**: TinyLlama 1.1B Chat (~2.1 GB)
**Status**: ✅ Pronto para uso
**Testado**: ✅ Funcionando corretamente

---

## 🎯 Opções de Uso

### Opção 1: Testar o Modelo (Recomendado Primeiro)

```bash
python test_model.py
```

Isso irá:
- ✅ Validar que o modelo está instalado
- ✅ Testar 3 gerações de texto
- ✅ Verificar dependências

**Tempo estimado**: 30-60 segundos

---

### Opção 2: Executar a Aplicação Web

```bash
python main.py
```

Acesse: http://localhost:8000

**Funcionalidades**:
- 💬 Chat interativo
- 🌐 Interface web moderna
- 📊 Monitoramento do modelo

---

### Opção 3: Usar Programaticamente

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Carregar modelo
model_path = "models/tinyllama-1.1b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Gerar resposta
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

### Opção 4: Docker

```bash
docker-compose up --build
```

Acesse: http://localhost:8000

---

## 📦 Baixar Outro Modelo (Opcional)

```bash
python download_model.py
```

**Modelos disponíveis**:
- TinyLlama 1.1B ✅ (já instalado)
- Open LLaMA 3B (maior, melhor qualidade)
- Open LLaMA 7B (ainda maior)
- Meta LLaMA 2/3 (requer login no HuggingFace)

---

## 🔧 Comandos Úteis

### Verificar Status
```bash
# Ver arquivos do projeto
ls -lah

# Ver modelos instalados
ls -lah models/

# Ver tamanho do modelo
du -sh models/tinyllama-1.1b-chat/
```

### Instalar Dependências (se necessário)
```bash
pip install -r requirements.txt
```

### Atualizar Dependências
```bash
pip install --upgrade transformers torch
```

---

## 🐛 Problemas Comuns

### "Module not found: transformers"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
```python
# Use CPU ao invés de GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu"
)
```

### Modelo muito lento
1. Use GPU se disponível
2. Reduza `max_length` nas gerações
3. Considere modelo menor (TinyLlama já é o menor)

---

## 📚 Documentação Completa

- **[README.md](README.md)** - Documentação principal
- **[models/README.md](models/README.md)** - Guia de uso dos modelos
- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Resumo da refatoração
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Solução de problemas

---

## 🎓 Exemplos de Uso

### Chat Simples
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/tinyllama-1.1b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        break

    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"AI: {response}\n")
```

### Com LangChain
```python
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_path = "models/tinyllama-1.1b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)
llm = HuggingFacePipeline(pipeline=pipe)

# Usar
response = llm.invoke("Explain AI in simple terms")
print(response)
```

### API REST
```bash
# Iniciar servidor
python main.py

# Em outro terminal, testar:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

---

## ✨ Próximos Passos

1. ✅ **Testar**: `python test_model.py`
2. ✅ **Executar**: `python main.py`
3. ✅ **Explorar**: Abra http://localhost:8000
4. ✅ **Personalizar**: Edite `config.py` conforme necessário
5. ✅ **Documentar**: Leia [README.md](README.md) para recursos avançados

---

## 💡 Dicas

- **Performance**: Use GPU se disponível (10-50x mais rápido)
- **Qualidade**: Modelos maiores = respostas melhores (mas mais lentos)
- **Memória**: TinyLlama usa ~2.5GB RAM, modelos maiores usam muito mais
- **Tokens**: Mais tokens = respostas mais longas (mas mais lentas)

---

**Pronto para começar!** 🎉

Execute `python test_model.py` para validar tudo!
