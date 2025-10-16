# Modelos LLaMA

Este diretório contém modelos de linguagem LLaMA baixados do Hugging Face.

## Modelo Atual: TinyLlama 1.1B Chat

**TinyLlama** é um modelo open-source compatível com a arquitetura LLaMA, otimizado para chat e geração de texto.

### Especificações
- **Nome**: TinyLlama-1.1B-Chat-v1.0
- **Parâmetros**: 1.1 bilhões
- **Arquitetura**: LLaMA-compatível
- **Tamanho**: ~2.2 GB
- **Contexto**: 2048 tokens
- **Licença**: Apache 2.0 (completamente open-source)

### Como usar no Python

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Carregar modelo e tokenizer
model_path = "models/tinyllama-1.1b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Gerar texto
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Como usar com LangChain

```python
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "models/tinyllama-1.1b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)
response = llm.invoke("Explain quantum computing")
print(response)
```

### Como usar com llama.cpp (mais rápido)

Para melhor performance, você pode converter o modelo para formato GGUF e usar com llama.cpp:

```bash
# Instalar llama.cpp
pip install llama-cpp-python

# Converter modelo (opcional, mas recomendado para performance)
# Baixar: https://github.com/ggerganov/llama.cpp
```

```python
from llama_cpp import Llama

# Carregar modelo
llm = Llama(
    model_path="models/tinyllama-1.1b-chat/model.gguf",
    n_ctx=2048,
    n_threads=4
)

# Gerar texto
output = llm("Q: What is AI? A:", max_tokens=200)
print(output['choices'][0]['text'])
```

## Outros Modelos Disponíveis

Se você quiser baixar modelos LLaMA oficiais da Meta, você precisará:

1. Criar conta no Hugging Face: https://huggingface.co/join
2. Aceitar termos de uso do modelo em: https://huggingface.co/meta-llama
3. Fazer login: `huggingface-cli login`
4. Executar: `python download_llama.py`

### Modelos Recomendados:

**Modelos Open-Source (sem autenticação):**
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - 1.1B (atual)
- `openlm-research/open_llama_3b` - 3B parâmetros
- `openlm-research/open_llama_7b` - 7B parâmetros

**Modelos Oficiais Meta (requer autenticação):**
- `meta-llama/Llama-2-7b-hf` - 7B base
- `meta-llama/Llama-2-7b-chat-hf` - 7B chat
- `meta-llama/Llama-3-8B` - 8B (mais recente)
- `meta-llama/Llama-3.1-8B` - 8B (ainda mais recente)

## Requisitos

```bash
pip install transformers torch accelerate huggingface_hub
```

Para melhor performance (opcional):
```bash
pip install bitsandbytes  # Para quantização
pip install llama-cpp-python  # Para inferência rápida
```

## Troubleshooting

### Erro: Out of Memory
- Use `torch_dtype=torch.float16` ao carregar o modelo
- Use `device_map="auto"` para distribuir entre GPU/CPU
- Considere usar quantização com bitsandbytes

### Modelo lento
- Use llama.cpp para inferência mais rápida
- Ative quantização (int8 ou int4)
- Reduza batch size e max_length

### Erro de autenticação
- Execute: `huggingface-cli login`
- Ou defina: `export HUGGING_FACE_HUB_TOKEN=seu_token`

## Links Úteis

- Hugging Face: https://huggingface.co
- TinyLlama: https://github.com/jzhang38/TinyLlama
- LLaMA oficial: https://ai.meta.com/llama/
- Transformers docs: https://huggingface.co/docs/transformers
