"""
Configuration file for the ML File Analysis System
"""

import os

# GPU Configuration - ALWAYS GPU-ONLY
GPU_ONLY_MODE = True  # Always enforce GPU-only mode

# Model Configuration - Llama3 only
DEFAULT_MODEL_TYPE = 'llama3'  # Fixed to Llama3

# File functionality removed

# Server Configuration
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))

# PyTorch Configuration (Llama3)
TORCH_DEVICE = os.getenv('TORCH_DEVICE', 'auto')  # auto, cuda, cpu
TORCH_DTYPE = os.getenv('TORCH_DTYPE', 'auto')  # auto, float16, float32

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

def get_device_config():
    """Get device configuration - GPU-ONLY MODE (Llama3 only)"""
    import torch
    
    # Check GPU availability
    pytorch_cuda_available = torch.cuda.is_available()
    
    # GPU-ONLY MODE: Fail if no GPU available
    if not pytorch_cuda_available:
        raise RuntimeError("GPU-ONLY MODE: CUDA is not available. GPU is required for Llama3.")
    
    config = {
        'gpu_only_mode': True,  # Always true
        'pytorch_cuda_available': pytorch_cuda_available,
        'device': 'cuda'  # Always CUDA
    }
    
    return config

def print_config():
    """Print current configuration"""
    print("🔧 Llama3 File Analysis System Configuration - GPU-ONLY MODE")
    print("=" * 60)
    print(f"🚀 GPU-Only Mode: {GPU_ONLY_MODE} (ENFORCED)")
    print(f"Model: {DEFAULT_MODEL_TYPE} (Fixed)")
    print(f"Server: {HOST}:{PORT}")
    print(f"PyTorch Device: {TORCH_DEVICE}")
    print(f"PyTorch Data Type: {TORCH_DTYPE}")
    print(f"Log Level: {LOG_LEVEL}")
    
    try:
        device_config = get_device_config()
        print("\n🎮 GPU-ONLY Device Configuration (Llama3):")
        print(f"  PyTorch CUDA Available: {device_config['pytorch_cuda_available']}")
        print(f"  Device: {device_config['device']} (GPU-ONLY)")
        print("  ✅ GPU-ONLY mode validated successfully for Llama3!")
    except Exception as e:
        print(f"\n❌ GPU-ONLY Configuration Error: {e}")
        print("   This system requires a CUDA-enabled GPU to run Llama3.")

if __name__ == "__main__":
    print_config()
