# Correção: Respostas Duplicadas no Histórico

## Problema Identificado

Quando o usuário fazia múltiplas perguntas em sequência, o modelo retornava:
- A resposta anterior completa
- MAIS a resposta atual
- Resultando em respostas cada vez mais longas e repetitivas

**Exemplo do problema:**
```
Pergunta 1: "O que é Python?"
Resposta: "Python é uma linguagem..."

Pergunta 2: "Como instalar Python?"
Resposta: "Python é uma linguagem... Como instalar Python..."
                ^^^^^ resposta anterior repetida
```

## Causa Raiz

O problema estava em **[main.py:895-898](file://c:/Users/jpedr/Documents/Programação/ML/main.py#L895-L898)**:

### Código ANTIGO (❌ Problema):
```python
# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Remove the input prompt from the response
response = response.replace(prompt, "").strip()
```

**Por que causava o problema:**

1. `outputs[0]` contém **TODA** a sequência: `[input_prompt + resposta_gerada]`
2. O `tokenizer.decode(outputs[0])` decodificava tudo, incluindo:
   - O histórico da conversa
   - Contexto RAG
   - A pergunta atual
   - A resposta gerada

3. A tentativa de remover com `response.replace(prompt, "")` falhava porque:
   - O `prompt` é a string exata antes da tokenização
   - Após decodificar, a formatação pode mudar ligeiramente
   - Alguns tokens especiais podem não ser removidos corretamente
   - O histórico continuava aparecendo na resposta

## Solução Implementada

### Código NOVO (✅ Correto):
```python
# Decode ONLY the new tokens (not including the input prompt)
# outputs[0] contains: [input_ids + generated_ids]
# We want only the generated part
input_length = inputs['input_ids'].shape[1]
generated_tokens = outputs[0][input_length:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
```

**Como funciona:**

1. Captura o tamanho do input: `input_length = inputs['input_ids'].shape[1]`
   - Isso conta quantos tokens foram no prompt original

2. Extrai APENAS os tokens gerados: `generated_tokens = outputs[0][input_length:]`
   - Faz um slice do tensor pegando apenas os novos tokens
   - `[input_length:]` significa "do índice input_length até o final"

3. Decodifica apenas a resposta nova: `tokenizer.decode(generated_tokens)`
   - Agora decodifica apenas os tokens que o modelo gerou
   - Sem histórico, sem prompt, apenas a resposta pura

## Visualização

```
╔════════════════════════════════════════════════════════════╗
║                    outputs[0]                              ║
╠════════════════════════════════════════════════════════════╣
║  Input Tokens (prompt)      │  Generated Tokens (resposta)║
║                              │                             ║
║  [histórico + pergunta]      │  [resposta do modelo]      ║
║                              │                             ║
║  0 ... input_length-1        │  input_length ... end      ║
║                              │                             ║
║  ❌ NÃO queremos isso        │  ✅ Queremos APENAS isso   ║
╚══════════════════════════════╧═════════════════════════════╝
```

## Teste

### Antes da correção (❌):
```
User: O que é RAG?
Assistant: RAG é Retrieval-Augmented Generation...

User: Como funciona?
Assistant: RAG é Retrieval-Augmented Generation... Como funciona? RAG funciona buscando...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           (repetiu a resposta anterior)
```

### Depois da correção (✅):
```
User: O que é RAG?
Assistant: RAG é Retrieval-Augmented Generation...

User: Como funciona?
Assistant: RAG funciona buscando documentos relevantes em um banco vetorial...
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           (apenas a resposta nova)
```

## Arquivos Modificados

- **[main.py:894-905](file://c:/Users/jpedr/Documents/Programação/ML/main.py#L894-L905)** - Função `chat_with_huggingface_model()`
  - Alterado para decodificar apenas tokens gerados
  - Removida a tentativa falha de usar `str.replace()`

## Impacto

✅ **Benefícios:**
- Respostas limpas sem repetição do histórico
- Conversas mais naturais e fluídas
- Menor uso de tokens na resposta
- Melhor experiência do usuário

⚠️ **Notas:**
- Esta correção afeta apenas o modelo Hugging Face
- Os modelos GGUF e Ollama já funcionavam corretamente
- O TRM não foi afetado pois usa outra implementação

## Detalhes Técnicos

### Por que `str.replace()` não funcionava?

1. **Diferenças de tokenização:**
   ```python
   # Antes da tokenização
   prompt = "User: Hello\nAssistant:"

   # Após decodificação (pode ter mudanças sutis)
   decoded = "User: Hello \n Assistant:"
            # ^^^ espaços extras podem aparecer
   ```

2. **Tokens especiais:**
   - Alguns tokens especiais não são completamente removidos
   - `<|begin_of_text|>`, `<|eot_id|>` podem aparecer malformados

3. **Problema com histórico longo:**
   - Com histórico de 5+ mensagens, a string fica muito longa
   - `str.replace()` é O(n*m) - muito lento
   - Pode falhar parcialmente deixando fragmentos

### Por que a nova solução é melhor?

1. **Matemática correta:**
   ```python
   # Sabemos exatamente onde começa a resposta
   input_length = 150  # exemplo: prompt tinha 150 tokens
   outputs[0] = [1, 2, 3, ..., 150, 151, 152, ..., 200]
                                     ^^^^^^^^^^^^^^^^
                                     apenas estes tokens
   ```

2. **Performance:**
   - Slice de tensor é O(1) - instantâneo
   - Sem necessidade de busca de strings

3. **Confiável:**
   - Sempre funciona independente do conteúdo
   - Não depende de formatação de strings

## Referências

- Transformers `generate()`: https://huggingface.co/docs/transformers/main_classes/text_generation
- Token IDs e decodificação: https://huggingface.co/docs/transformers/main_classes/tokenizer

## Comandos para Testar

1. **Reinicie o servidor:**
   ```bash
   python main.py
   ```

2. **Faça perguntas em sequência:**
   - Pergunta 1: "O que é banco de dados vetorial?"
   - Pergunta 2: "Como funciona?"
   - Pergunta 3: "Quais as vantagens?"

3. **Verifique que:**
   - ✅ Cada resposta é única e independente
   - ✅ Não há repetição de respostas anteriores
   - ✅ O histórico é mantido (contexto), mas não aparece na resposta
