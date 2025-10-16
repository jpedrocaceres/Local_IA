# Sistema de Chat Multi-Modelo - GPU Only

Sistema de chat com múltiplos modelos de IA: Llama 3.1 8B e TRM (Tiny Recursive Models), usando PyTorch com processamento exclusivo em GPU e interface web moderna.

## 🎯 Modelos Disponíveis

### Llama 3.1 8B Instruct
✅ **Parâmetros**: 8 bilhões
✅ **Tamanho**: ~16 GB (FP16)
✅ **Uso**: Chat geral, instruções, conversação
✅ **VRAM**: 8-16GB

### TRM (Tiny Recursive Models)
✅ **Parâmetros**: 7 milhões
✅ **Tamanho**: ~28 MB
✅ **Uso**: Raciocínio, puzzles, tarefas lógicas
✅ **VRAM**: 2-4GB

## 🔄 Seleção de Modelo

Você pode escolher qual modelo usar antes de iniciar o servidor:

```bash
python select_model.py
```

## 🚀 Funcionalidades

- 🦙 **Llama 3.1 8B**: Modelo de linguagem avançado da Meta
- 🧠 **TRM**: Modelo recursivo compacto para raciocínio
- 🔄 **Seleção de Modelo**: Escolha o modelo antes de executar
- 💬 **Chat Interativo**: Interface web moderna e responsiva
- ⚡ **GPU Only**: Processamento exclusivo em GPU NVIDIA
- 🔧 **Múltiplos Backends**: Suporta Hugging Face, GGUF, Ollama e TRM
- 📦 **Extensível**: Arquitetura modular e configurável

## 📋 Pré-requisitos

### Obrigatórios
- Python 3.10+
- **NVIDIA GPU com CUDA 11.8+** (obrigatório)
- 16GB+ VRAM (GPU)
- 32GB+ RAM (sistema)
- 20GB+ espaço em disco

### Software
- CUDA Toolkit 11.8+
- cuDNN 8.x
- Docker (opcional)

## 🏁 Início Rápido

### 1. Verificar GPU

```bash
# Verificar se CUDA está disponível
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt
```

### 3. Selecionar Modelo

```bash
python select_model.py
```

Escolha entre:
- **[1] Llama 3.1 8B** - Para chat geral e instruções
- **[2] TRM** - Para tarefas de raciocínio e puzzles

### 4. Configurar o Modelo Escolhido

#### Se escolheu Llama 3.1 (escolha uma opção)

**Opção A: Ollama (Recomendado)**
```bash
# Instalar Ollama: https://ollama.ai
ollama pull llama3.1:8b
```

**Opção B: Hugging Face**
```bash
# 1. Login no Hugging Face
huggingface-cli login

# 2. Aceitar termos em: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

# 3. Baixar modelo
python download_model.py
```

#### Se escolheu TRM

```bash
# 1. Executar setup do TRM
python setup_trm.py

# 2. Treinar modelo (ou baixar checkpoint pré-treinado)
cd models/TRM
python pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]"
```

Para mais detalhes sobre TRM, veja [TRM_GUIDE.md](TRM_GUIDE.md)

### 5. Executar a Aplicação

```bash
python main.py
```

Acesse: http://localhost:8000

## 🔧 Configuração

### Variáveis de Ambiente

Crie um arquivo `.env`:

```env
# GPU (obrigatório)
CUDA_VISIBLE_DEVICES=0
GPU_ONLY_MODE=true

# Modelo
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
MODEL_PATH=models/llama3.1-8b

# API
PORT=8000
HOST=0.0.0.0
```

### Backends Suportados

O sistema detecta automaticamente o backend disponível:

1. **GGUF** - Se encontrar arquivo `.gguf` no diretório do modelo
2. **Ollama** - Se Ollama estiver rodando em `http://localhost:11434`
3. **Hugging Face** - Fallback padrão

## 📦 Modelos

### Llama 3.1 8B Instruct

**Especificações:**
- Parâmetros: 8 bilhões
- Tamanho: ~16 GB (FP16)
- Contexto: 128K tokens (ajustável)
- Licença: Llama 3.1 Community License
- Uso: Chat, instruções, reasoning

**Links:**
- Hugging Face: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Ollama: `llama3.1:8b`

## 💻 Uso Programático

### Básico com Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"  # GPU automático
)

# Gerar texto
prompt = "What is artificial intelligence?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Com Ollama

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.1:8b",
        "prompt": "Explain quantum computing",
        "stream": False
    }
)
print(response.json()["response"])
```

### Com GGUF

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/llama3.1-8b/model.gguf",
    n_gpu_layers=-1,  # Usar todas as camadas na GPU
    n_ctx=8192
)

output = llm(
    "Hello!",
    max_tokens=256,
    temperature=0.7
)
print(output["choices"][0]["text"])
```

## 🐳 Docker

### Opção 1: Docker Compose (Recomendado)

```bash
docker-compose up --build
```

### Opção 2: Docker Manual

```bash
docker build -t llama-chat .
docker run --gpus all -p 8000:8000 -v ./models:/app/models llama-chat
```

**Nota**: Docker requer NVIDIA Container Toolkit para suporte a GPU.

## 📚 API Endpoints

### Health Check
```bash
GET /api/health

Response:
{
  "status": "healthy",
  "mode": "GPU_ONLY",
  "model_loaded": true,
  "model_type": "ollama",
  "device": "cuda",
  "cuda_available": true,
  "gpu_info": {
    "device_name": "NVIDIA GeForce RTX 3090",
    "memory_total": "24.0 GB",
    "memory_allocated": "15.2 GB",
    "memory_cached": "16.1 GB"
  }
}
```

### Chat
```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Explain quantum computing in simple terms"
}

Response:
{
  "response": "Quantum computing is...",
  "status": "success"
}
```

## 📁 Estrutura do Projeto

```
ML/
├── models/                      # Modelos LLaMA
│   ├── llama3.1-8b/            # Llama 3.1 8B (local)
│   └── README.md               # Guia de modelos
├── mcp/                         # MCP Server
│   └── __init__.py
├── static/                      # Interface web
│   └── index.html
├── main.py                      # Aplicação FastAPI
├── config.py                    # Configurações GPU
├── setup.py                     # Setup inicial
├── download_model.py            # Download de modelos
├── test_model.py                # Testes
├── requirements.txt             # Dependências Python
├── Dockerfile                   # Docker config
├── docker-compose.yml           # Docker Compose
├── QUICKSTART.md                # Guia rápido
├── TROUBLESHOOTING.md           # Solução de problemas
└── README.md                    # Este arquivo
```

## ⚡ Otimizações

### 1. Quantização 4-bit (Reduz VRAM)

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 2. Flash Attention 2 (Mais Rápido)

```bash
pip install flash-attn --no-build-isolation
```

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
```

### 3. Compilação (PyTorch 2.0+)

```python
import torch

model = torch.compile(model)  # Acelera inferência
```

## 🐛 Troubleshooting

### GPU não detectada
```bash
# Verificar CUDA
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Reinstalar PyTorch com CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (OOM)
**Soluções:**
1. Use quantização 4-bit (ver Otimizações)
2. Reduza `max_new_tokens` em gerações
3. Use modelo GGUF com menos camadas na GPU
4. Feche outros programas usando VRAM

### Ollama não conecta
```bash
# Verificar se Ollama está rodando
curl http://localhost:11434/api/tags

# Iniciar Ollama
ollama serve

# Verificar modelo instalado
ollama list
```

### Modelo lento
**Soluções:**
1. Use Ollama (otimizado)
2. Ative Flash Attention 2
3. Reduza contexto (`max_length`)
4. Use compilação PyTorch

## 🔄 Trocar de Modelo

Para alternar entre Llama 3.1 e TRM:

1. **Parar o servidor** (Ctrl+C)
2. **Executar**: `python select_model.py`
3. **Selecionar novo modelo**
4. **Reiniciar**: `python main.py`

## 📖 Documentação Adicional

- [TRM_GUIDE.md](TRM_GUIDE.md) - Guia completo do TRM
- [QUICKSTART.md](QUICKSTART.md) - Guia de início rápido
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Solução de problemas
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [TRM Repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [Transformers Docs](https://huggingface.co/docs/transformers)
- [Ollama Docs](https://ollama.ai/docs)

## 🔒 Segurança

- A aplicação roda em `0.0.0.0:8000` por padrão
- Para produção, adicione autenticação (JWT, OAuth)
- Use HTTPS em produção (nginx + certbot)
- Valide e sanitize inputs do usuário
- Implemente rate limiting
- Monitore uso de GPU e custos

## 📝 Licença

- Projeto: MIT License
- Llama 3.1: [Llama 3.1 Community License](https://ai.meta.com/llama/license/)

**Importante**: Revise e aceite os termos de uso do Llama 3.1 antes de usar em produção.

## 🔌 MCP Server (Model Context Protocol)

O projeto inclui um **servidor MCP** com ferramentas especializadas para agronegócio que podem ser usadas pelo Claude Desktop e outros clientes MCP.

### Ferramentas Disponíveis

#### 🌤️ Previsão do Tempo Agrícola
Consulta previsão do tempo com recomendações específicas para agricultura.

```python
# Exemplo via Claude Desktop
"Qual a previsão do tempo para Corumbá, MS para os próximos 7 dias?"
```

#### 💰 Preços de Commodities
Consulta preços de soja, milho, café, boi gordo e outras commodities com análise de mercado.

```python
# Exemplo via Claude Desktop
"Qual o preço atual da soja e me dê recomendações de comercialização?"
```

#### 📅 Calendário Agrícola
Fornece informações sobre épocas de plantio e colheita regionalizadas.

```python
# Exemplo via Claude Desktop
"Quando plantar soja no Centro-Oeste? Estamos em qual fase do calendário?"
```

### Configuração do MCP

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar Claude Desktop:**

   Edite `%APPDATA%\Claude\claude_desktop_config.json` (Windows) ou `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

   ```json
   {
     "mcpServers": {
       "pantanal-agricola": {
         "command": "python",
         "args": ["C:\\caminho\\completo\\mcp_server.py"]
       }
     }
   }
   ```

3. **Reiniciar Claude Desktop**

📖 **Guia Completo:** [MCP_SETUP.md](MCP_SETUP.md)

## 💡 Roadmap

- [x] ✅ Suporte a múltiplos modelos (Llama + TRM)
- [x] ✅ Sistema de seleção de modelo interativo
- [x] ✅ Integração TRM para raciocínio
- [ ] Suporte a streaming de respostas
- [ ] Sistema de memória/contexto persistente
- [ ] Interface web para seleção de modelo
- [ ] Integração com pgvector para RAG
- [ ] API de embeddings
- [ ] Multi-GPU support
- [ ] Quantização automática baseada em VRAM
- [ ] Ensemble de modelos (Llama + TRM)

## 📞 Suporte

Problemas comuns e soluções:

1. Verifique logs: `python main.py` ou `docker-compose logs`
2. Teste GPU: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verifique health: `curl http://localhost:8000/api/health`
4. Consulte [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
5. Teste modelo: `python test_model.py`

---

**Made with ❤️ using Llama 3.1 and Transformers**
