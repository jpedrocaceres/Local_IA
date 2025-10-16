# Exemplos de Uso - Multi-Modelo

Este guia mostra exemplos práticos de como usar cada modelo no sistema.

## 🦙 Exemplos com Llama 3.1

### 1. Conversação Geral

**Bom para**: Chat casual, perguntas gerais, explicações

```
Usuário: Olá! Como você está?
Llama: Olá! Estou funcionando bem, obrigado por perguntar. Como posso ajudá-lo hoje?

Usuário: Explique o que é Machine Learning
Llama: Machine Learning é um subcampo da inteligência artificial que permite
que computadores aprendam e melhorem a partir de experiências sem serem
explicitamente programados...
```

### 2. Geração de Código

**Bom para**: Programação, scripts, debugging

```
Usuário: Escreva uma função Python para calcular fatorial
Llama:
def fatorial(n):
    if n == 0 or n == 1:
        return 1
    return n * fatorial(n - 1)

# Exemplo de uso
print(fatorial(5))  # Output: 120
```

### 3. Análise de Documentos

**Bom para**: Resumos, análise de textos, Q&A sobre documentos

```
Usuário: [Upload PDF] Resuma este documento em 3 pontos
Llama: Com base no documento fornecido:
1. O documento trata sobre...
2. Os principais achados incluem...
3. As conclusões indicam que...
```

### 4. Criação Criativa

**Bom para**: Histórias, poemas, roteiros

```
Usuário: Escreva um haiku sobre programação
Llama:
Código compila
Bugs se escondem no silêncio
Debug até tarde
```

---

## 🧠 Exemplos com TRM

### 1. Problemas Lógicos

**Bom para**: Raciocínio dedutivo, relações lógicas

```
Usuário: Se A > B e B > C, qual a relação entre A e C?
TRM: [Iteração 1] Analisando relações...
[Iteração 2] A é maior que B, B é maior que C
[Iteração 3] Por transitividade, A > C
Resposta final: A é maior que C (A > C)
```

### 2. Reconhecimento de Padrões

**Bom para**: Sequências, padrões numéricos

```
Usuário: Complete a sequência: 2, 4, 8, 16, ?
TRM: [Iteração 1] Identificando padrão...
[Iteração 2] Cada número é o dobro do anterior
[Iteração 3] 16 × 2 = 32
Resposta: 32

Padrão detectado: Progressão geométrica com razão 2
```

### 3. Problemas de Labirinto

**Bom para**: Pathfinding, navegação espacial

```
Usuário: Em um labirinto 5x5, encontre o caminho de (0,0) até (4,4)
TRM: [Iteração 1] Mapeando possíveis caminhos...
[Iteração 2] Eliminando becos sem saída...
[Iteração 3] Otimizando rota...
[Iteração 4] Verificando solução...

Caminho encontrado:
(0,0) → (0,1) → (1,1) → (2,1) → (2,2) → (3,2) → (4,2) → (4,3) → (4,4)
Passos totais: 8
```

### 4. Puzzles tipo ARC-AGI

**Bom para**: Transformações visuais, regras de padrões

```
Usuário: Qual é a próxima figura na sequência? [descrição do padrão]
TRM: [Iteração 1] Analisando padrão visual...
[Iteração 2] Identificando regra de transformação...
[Iteração 3] Aplicando regra à próxima posição...
[Iteração 4] Refinando resultado...
[Iteração 5] Validando consistência...

Regra identificada: Rotação de 90° no sentido horário + espelhamento
Próxima figura: [descrição da solução]
```

---

## 🔄 Quando Trocar de Modelo

### Use Llama 3.1 quando:
- Precisar de **conhecimento geral** amplo
- Quiser **conversação natural** e fluida
- Necessitar **geração de texto longo**
- Trabalhar com **contexto complexo**
- Precisar de **criatividade** e variação

### Use TRM quando:
- Trabalhar com **puzzles lógicos**
- Precisar de **raciocínio estruturado**
- Resolver **problemas matemáticos**
- Trabalhar com **padrões** e sequências
- Ter **baixa VRAM** disponível
- Precisar de **refinamento iterativo**

---

## 📊 Comparação Prática

### Pergunta: "Resolva este Sudoku..."

**Llama 3.1:**
```
Resposta: Vou tentar resolver este Sudoku passo a passo...
[Dá uma solução, mas pode ter erros em casos complexos]
```

**TRM:**
```
[Iteração 1] Analisando restrições...
[Iteração 2] Aplicando lógica de eliminação...
[Iteração 3] Propagando valores conhecidos...
[Iteração 4] Refinando solução...
[Iteração 5-10] Validando e ajustando...
Resposta: [Solução mais precisa com processo iterativo]
```

### Pergunta: "Escreva uma história sobre IA"

**Llama 3.1:**
```
Era uma vez, em um futuro não muito distante, uma IA chamada ARIA...
[História completa e criativa com 500+ palavras]
```

**TRM:**
```
[Resposta limitada - não é o forte do TRM]
História básica sobre IA... [texto curto e menos criativo]
```

---

## 🎯 Dicas de Uso

### Para Llama 3.1

1. **Seja específico**: "Explique X em termos simples para um iniciante"
2. **Use contexto**: Forneça documentos e arquivos para análise
3. **Conversação**: Mantenha histórico para contexto
4. **Exemplos**: Peça exemplos práticos

### Para TRM

1. **Problemas estruturados**: Formule como puzzle ou problema lógico
2. **Seja claro**: Defina regras e restrições claramente
3. **Paciência**: TRM itera várias vezes (mais lento, mais preciso)
4. **Validação**: TRM mostra processo de raciocínio

---

## 🚀 Workflow Sugerido

### Scenario 1: Desenvolvimento de Software

1. **Use Llama** para:
   - Planejar arquitetura
   - Gerar código inicial
   - Documentação

2. **Use TRM** para:
   - Debug de lógica complexa
   - Otimização de algoritmos
   - Análise de fluxo de controle

### Scenario 2: Análise de Dados

1. **Use Llama** para:
   - Interpretar resultados
   - Gerar relatórios
   - Sugerir análises

2. **Use TRM** para:
   - Identificar padrões nos dados
   - Validar correlações lógicas
   - Resolver problemas de otimização

### Scenario 3: Educação

1. **Use Llama** para:
   - Explicar conceitos
   - Gerar exemplos
   - Tutoria geral

2. **Use TRM** para:
   - Resolver exercícios de lógica
   - Problemas matemáticos
   - Puzzles educacionais

---

## 🔗 API Examples

### Python Client

```python
import requests

# Chat com modelo ativo (Llama ou TRM)
def chat(message, history=[]):
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={
            "message": message,
            "history": history
        }
    )
    return response.json()

# Exemplo de uso
result = chat("Explain quantum computing")
print(result['response'])
```

### cURL

```bash
# Check which model is loaded
curl http://localhost:8000/api/health

# Send chat message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello!",
    "history": []
  }'
```

---

## 📖 Próximos Passos

1. **Experimente ambos os modelos**: Use `python select_model.py`
2. **Compare resultados**: Teste a mesma pergunta em ambos
3. **Escolha o melhor**: Baseado no seu caso de uso
4. **Feedback**: Reporte problemas e sugestões

Para mais informações:
- [README.md](README.md) - Documentação completa
- [TRM_GUIDE.md](TRM_GUIDE.md) - Guia específico do TRM
- [QUICK_START_MODELS.md](QUICK_START_MODELS.md) - Início rápido
