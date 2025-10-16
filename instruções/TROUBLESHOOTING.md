# Troubleshooting Guide

## Common Issues and Solutions

### 1. Virtual Environment Creation Failed

**Error**: `CREATE_VENV.PIP_FAILED_INSTALL_REQUIREMENTS`

**Solutions**:

#### Option A: Use the setup script
```bash
python setup-local.py
```

#### Option B: Manual setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### Option C: Use Docker (Recommended)
```bash
docker-compose up --build
```

### 2. Package Installation Conflicts

**Error**: Package version conflicts or installation failures

**Solutions**:

1. **Use the requirements file**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install packages individually**:
   ```bash
   pip install fastapi uvicorn pydantic python-multipart aiofiles
   pip install transformers torch accelerate sentencepiece protobuf
   ```

3. **Use conda instead of pip**:
   ```bash
   conda create -n llama3-env python=3.9
   conda activate llama3-env
   pip install -r requirements.txt
   ```

### 3. CUDA/GPU Issues

**Error**: CUDA not available or GPU not detected

**Solutions**:

1. **Check NVIDIA drivers**:
   ```bash
   nvidia-smi
   ```

2. **Install NVIDIA Container Toolkit** (for Docker):
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Test GPU support**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

### 4. Model Download Issues

**Error**: Model download fails or times out

**Solutions**:

1. **Increase timeout**:
   ```python
   # In main.py, modify the model loading section
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16,
       device_map="auto",
       low_cpu_mem_usage=True,
       resume_download=True  # Add this line
   )
   ```

2. **Use a different model**:
   ```python
   # Change model_name in main.py
   model_name = "microsoft/DialoGPT-medium"  # Smaller model for testing
   ```

3. **Download model manually**:
   ```bash
   # Download to local directory
   git lfs install
   git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
   ```

### 5. Memory Issues

**Error**: Out of memory (OOM) errors

**Solutions**:

1. **Use CPU instead of GPU**:
   ```python
   # In main.py, force CPU usage
   device = "cpu"
   model = model.to(device)
   ```

2. **Reduce model size**:
   ```python
   # Use a smaller model
   model_name = "microsoft/DialoGPT-small"
   ```

3. **Optimize memory usage**:
   ```python
   # In main.py, add memory optimization
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16,
       device_map="auto",
       low_cpu_mem_usage=True,
       max_memory={0: "10GB"}  # Limit GPU memory
   )
   ```

### 6. Port Already in Use

**Error**: Port 8000 is already in use

**Solutions**:

1. **Change port in docker-compose.yml**:
   ```yaml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

2. **Kill process using port 8000**:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   
   # Linux/Mac
   lsof -ti:8000 | xargs kill -9
   ```

### 7. File Upload Issues

**Error**: File upload fails or files not accessible

**Solutions**:

1. **Check file permissions**:
   ```bash
   chmod 755 files/
   ```

2. **Check file size limits**:
   - Default limit is 10MB
   - Modify in main.py if needed

3. **Check file format**:
   - Currently supports text files
   - Add support for other formats in main.py

### 8. Slow Performance

**Issues**: Model responses are slow

**Solutions**:

1. **Use GPU acceleration**:
   - Ensure CUDA is properly installed
   - Check GPU memory usage

2. **Optimize model parameters**:
   ```python
   # Reduce max_new_tokens for faster responses
   outputs = model.generate(
       **inputs,
       max_new_tokens=256,  # Reduced from 512
       temperature=0.7,
       do_sample=True,
       pad_token_id=tokenizer.eos_token_id
   )
   ```

3. **Use a smaller model**:
   - Switch to a smaller Llama model
   - Or use a different model entirely

## Getting Help

If you're still experiencing issues:

1. **Check the logs**:
   ```bash
   docker-compose logs
   ```

2. **Test individual components**:
   ```bash
   # Test API health
   curl http://localhost:8000/api/health
   
   # Test file listing
   curl http://localhost:8000/api/files
   ```

3. **Run in debug mode**:
   ```bash
   # Add debug logging to main.py
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Check system requirements**:
   - Python 3.8+
   - 16GB+ RAM
   - NVIDIA GPU (recommended)
   - Docker (for containerized deployment)

## Alternative Deployment Options

### Option 1: Local Python (No Docker)
```bash
python setup-local.py
python main.py
```

### Option 2: Docker without GPU
```bash
# Modify docker-compose.yml to remove GPU requirements
docker-compose up --build
```

### Option 3: Use a different model
```python
# In main.py, change the model name
model_name = "microsoft/DialoGPT-medium"  # Smaller, faster model
```
