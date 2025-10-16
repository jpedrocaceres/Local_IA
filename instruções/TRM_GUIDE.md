# TRM (Tiny Recursive Models) Integration Guide

## Overview

This guide explains how to use TRM (Tiny Recursive Models) as an alternative to Llama 3.1 in this chat application.

**TRM** is a tiny 7M parameter neural network developed by Samsung SAIL Montreal that recursively improves predicted answers through iterative refinement.

## Key Features

- **Tiny Size**: Only 7M parameters (~28MB)
- **Recursive Reasoning**: Iteratively refines answers up to K steps
- **Specialized Tasks**: Excels at reasoning tasks (ARC-AGI, Sudoku, Mazes)
- **Low VRAM**: Requires only 2-4GB VRAM vs 16GB for Llama

## Performance

- **ARC-AGI-1**: 45% accuracy
- **ARC-AGI-2**: 8% accuracy
- **Sudoku-Extreme**: High accuracy
- **Maze-Hard**: High accuracy

## Requirements

### System Requirements
- Python 3.10+
- CUDA 12.6.0+ (GPU recommended)
- 2GB+ VRAM (GPU)
- 8GB+ RAM (system)
- 5GB+ disk space

### Software Dependencies
- PyTorch 2.0+
- einops, omegaconf, hydra-core
- numba, adam-atan2

## Installation

### Step 1: Select TRM Model

Run the model selection tool:

```bash
python select_model.py
```

Choose option `[2] TRM (Tiny Recursive Models)`

### Step 2: Install TRM

Run the setup script:

```bash
python setup_trm.py
```

This will:
1. Clone the TRM repository to `./models/TRM/`
2. Install TRM-specific dependencies
3. Create a Python wrapper for easy integration

### Step 3: Train or Download Model

#### Option A: Train Your Own Model

```bash
cd models/TRM

# Train on ARC-AGI dataset
python pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4
```

Training time: ~2-8 hours depending on GPU

#### Option B: Use Pre-trained Model (if available)

Check the [TRM repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) for pre-trained checkpoints.

Place checkpoint in one of these locations:
- `./models/TRM/checkpoints/best.pt`
- `./models/TRM/checkpoints/latest.pt`
- `./models/trm_checkpoint.pt`

### Step 4: Start the Server

```bash
python main.py
```

Access the interface at http://localhost:8000

## Usage

### Chat Interface

The TRM model works with the same chat interface as Llama 3.1:

```
User: Solve this sudoku puzzle: [puzzle here]
TRM: [Iteratively solves puzzle through recursive refinement]
```

### API Usage

#### Chat Endpoint

```bash
POST /api/chat
Content-Type: application/json

{
  "message": "Your reasoning task here",
  "history": []
}
```

Response:
```json
{
  "response": "TRM's answer after recursive refinement",
  "status": "success"
}
```

#### Health Check

```bash
GET /api/health
```

Response shows which model is loaded:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "trm",
  "device": "cuda"
}
```

## Switching Between Models

To switch between Llama 3.1 and TRM:

1. **Stop the server** (Ctrl+C)
2. **Run selection tool**: `python select_model.py`
3. **Choose model**: Select option 1 (Llama) or 2 (TRM)
4. **Restart server**: `python main.py`

## TRM Architecture

### Recursive Refinement Process

```
Input Question → Embedding → Initial Answer (y₀) + Latent (z₀)
                              ↓
                       Refinement Loop (K steps)
                              ↓
                    Update Latent (z₁, z₂, ..., zₖ)
                              ↓
                    Update Answer (y₁, y₂, ..., yₖ)
                              ↓
                         Final Answer (yₖ)
```

### Parameters

- **L_layers**: Number of transformer layers (default: 2)
- **H_cycles**: Number of horizontal cycles (default: 3)
- **L_cycles**: Number of latent cycles (default: 4)
- **max_iterations**: Recursive refinement steps (default: 10)

## Best Use Cases

### When to Use TRM

✅ **Reasoning tasks**: Logic puzzles, pattern recognition
✅ **Constrained problems**: Sudoku, mazes, ARC-AGI
✅ **Low VRAM**: Only 2-4GB needed
✅ **Fast inference**: Small model size
✅ **Iterative refinement**: Problems that benefit from multiple passes

### When to Use Llama 3.1

✅ **General chat**: Open-ended conversation
✅ **Instruction following**: Complex multi-step tasks
✅ **Long-form generation**: Articles, essays, code
✅ **Broad knowledge**: General knowledge questions
✅ **Context understanding**: Nuanced language tasks

## Configuration

### Adjusting TRM Parameters

Edit `trm_wrapper.py` to customize:

```python
model.generate(
    prompt=prompt,
    max_iterations=10,      # More iterations = better refinement
    temperature=0.7         # Higher = more creative
)
```

### Model Config

Edit `model_config.json` to set default model:

```json
{
  "selected_model": "trm",
  "models": {
    "trm": {
      "checkpoint_path": "./models/TRM/checkpoints/best.pt"
    }
  }
}
```

## Troubleshooting

### TRM Not Loading

**Error**: `TRM not installed. Please run: python setup_trm.py`

**Solution**:
```bash
python setup_trm.py
```

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'einops'`

**Solution**:
```bash
pip install einops omegaconf hydra-core numba adam-atan2
```

### No Trained Model

**Error**: `No model loaded. Train a model or provide checkpoint path.`

**Solution**:
1. Train a model following Step 3 above
2. Or download a pre-trained checkpoint
3. Place checkpoint in `./models/TRM/checkpoints/`

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
- TRM should only use 2-4GB VRAM
- Close other GPU programs
- Reduce batch size in training config

### Git Not Found

**Error**: `Failed to clone repository: git not found`

**Solution**:
1. Install Git: https://git-scm.com/
2. Or manually clone: `git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git models/TRM`

## Performance Comparison

| Feature | Llama 3.1 8B | TRM 7M |
|---------|--------------|--------|
| Parameters | 8 billion | 7 million |
| Size | ~16GB | ~28MB |
| Min VRAM | 8GB | 2GB |
| General Chat | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Reasoning Tasks | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | Medium | Fast |
| Setup | Easy (download) | Medium (train) |

## Advanced Topics

### Custom Training Data

To train TRM on your own data:

1. Format data following TRM format
2. Place in `models/TRM/data/`
3. Run training with custom data path:

```bash
python pretrain.py \
  arch=trm \
  data_paths="[data/your-custom-data]"
```

### Integration with RAG

TRM can be integrated with retrieval-augmented generation:

```python
# Retrieve relevant context
context = retriever.get_context(query)

# Add to TRM prompt
prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
response = model.generate(prompt)
```

### Ensemble Models

Run both models and combine outputs:

```python
# Use TRM for reasoning sub-tasks
reasoning_result = trm_model.generate(reasoning_query)

# Use Llama for final synthesis
final_answer = llama_model.generate(
    f"Given this reasoning: {reasoning_result}\n"
    f"Provide final answer to: {original_query}"
)
```

## Resources

- **TRM Repository**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- **Paper**: Check repository for research paper
- **Issues**: Report TRM-specific issues to Samsung SAIL Montreal
- **This Project**: Report integration issues here

## License

- **TRM**: Check TRM repository for license
- **This Integration**: MIT License

---

**Need Help?**

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Run model selection: `python select_model.py`
3. Verify installation: `python setup_trm.py`
4. Check health: `curl http://localhost:8000/api/health`
