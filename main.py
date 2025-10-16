from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import io
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from contextlib import asynccontextmanager
import json
import requests
import PyPDF2
from docx import Document
import pandas as pd

# Import configuration
from config import get_device_config

# PostgreSQL AI and Vector availability check
import importlib.util
POSTGRES_AI_AVAILABLE = (
    importlib.util.find_spec("pgai") is not None and
    importlib.util.find_spec("pgvector") is not None
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# RAG Database module - Using LlamaIndex
try:
    import rag_llamaindex
    RAG_DB_AVAILABLE = True
    logger.info("RAG LlamaIndex module loaded successfully")
except Exception as e:
    RAG_DB_AVAILABLE = False
    logger.warning(f"RAG LlamaIndex module not available: {e}")

# Global variables for models
model = None
tokenizer = None
device = None
model_type = None  # 'gguf', 'huggingface', 'ollama', or 'trm'
selected_model = None  # 'llama3.1' or 'trm'

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"

# File context storage
file_contexts = {}  # {session_id: {filename: content}}

def load_model_config():
    """Load model configuration from model_config.json"""
    config_path = "model_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"selected_model": "llama3.1"}

def configure_llamaindex_llm():
    """Configure LlamaIndex to use Llama 3.1 model"""
    global model, tokenizer

    try:
        from llama_index.core import Settings
        from llama_index.llms.huggingface import HuggingFaceLLM

        # Only configure once
        if hasattr(Settings, '_llm_configured'):
            return

        logger.info("🦙 Configuring LlamaIndex with Llama 3.1...")

        # Configure LlamaIndex to use our loaded model
        if model is not None and tokenizer is not None:
            Settings.llm = HuggingFaceLLM(
                context_window=8192,
                max_new_tokens=256,  # Reduzido para respostas concisas
                generate_kwargs={
                    "temperature": 0.3,  # Mais focado e menos criativo
                    "top_p": 0.85,  # Menos variação
                    "do_sample": True,
                    "repetition_penalty": 1.3,  # Penalizar repetições fortemente
                    "no_repeat_ngram_size": 3  # Evitar repetir frases
                },
                model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                device_map="auto",
                tokenizer_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
                model=model,
                tokenizer=tokenizer,
                system_prompt=(
                    "Você é um assistente especializado em dados de vendas agrícolas. "
                    "Responda SEMPRE de forma CONCISA e DIRETA. "
                    "Para perguntas simples, dê respostas curtas e objetivas. "
                    "NUNCA repita a mesma informação. "
                    "Seja breve e vá direto ao ponto. "
                    "Evite explicações longas ou redundantes."
                )
            )

            Settings._llm_configured = True
            logger.info("✓ LlamaIndex configured successfully with Llama 3.1!")
        else:
            logger.warning("⚠️  Model not loaded yet, using MockLLM for LlamaIndex")

    except Exception as e:
        logger.error(f"Failed to configure LlamaIndex LLM: {e}")
        logger.info("Continuing with MockLLM...")

class Message(BaseModel):
    role: str  # 'user' ou 'assistant'
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []  # Histórico de mensagens anteriores
    file_context: Optional[str] = None  # Contexto de arquivo opcional

class ChatResponse(BaseModel):
    response: str
    status: str

class FileUploadResponse(BaseModel):
    filename: str
    content_preview: str
    status: str
    message: str

class RAGIndexRequest(BaseModel):
    filename: str
    content: str
    metadata: Optional[dict] = None

class RAGIndexResponse(BaseModel):
    document_id: int
    num_chunks: int
    status: str
    message: str

class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    min_similarity: float = 0.5

class RAGSearchResponse(BaseModel):
    results: list
    status: str

class RAGStatsResponse(BaseModel):
    stats: dict
    status: str

# FileListResponse removed - no file operations

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar PDF: {str(e)}")

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from Word document"""
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOCX: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar DOCX: {str(e)}")

def extract_text_from_excel(file_content: bytes) -> str:
    """Extract text from Excel file"""
    try:
        df = pd.read_excel(io.BytesIO(file_content), sheet_name=None)
        text = ""
        for sheet_name, sheet_data in df.items():
            text += f"\n=== {sheet_name} ===\n"
            text += sheet_data.to_string(index=False) + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting Excel: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar Excel: {str(e)}")

def extract_text_from_csv(file_content: bytes) -> str:
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Error extracting CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar CSV: {str(e)}")

def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """Extract text from various file types"""
    file_extension = filename.lower().split('.')[-1]

    if file_extension == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(file_content)
    elif file_extension in ['xlsx', 'xls']:
        return extract_text_from_excel(file_content)
    elif file_extension == 'csv':
        return extract_text_from_csv(file_content)
    elif file_extension in ['txt', 'md', 'json', 'xml', 'html', 'py', 'js', 'java', 'cpp', 'c', 'h']:
        # Text files
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return file_content.decode('latin-1')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Erro ao decodificar arquivo de texto: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Tipo de arquivo não suportado: .{file_extension}")

def load_trm_model():
    """Load TRM (Tiny Recursive Models) model"""
    global model, tokenizer, device, model_type

    try:
        # Get device configuration
        device_config = get_device_config()
        device = device_config['device']  # Always 'cuda'

        logger.info(f"🚀 Loading TRM model on device: {device}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Check if TRM wrapper is available
        try:
            from trm_wrapper import TRMModel, check_trm_available

            if not check_trm_available():
                raise RuntimeError(
                    "TRM not installed. Please run: python setup_trm.py"
                )

            # Load TRM model
            model_path = None
            trm_checkpoint_paths = [
                "./models/TRM/checkpoints/best.pt",
                "./models/TRM/checkpoints/latest.pt",
                "./models/trm_checkpoint.pt"
            ]

            for path in trm_checkpoint_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found TRM checkpoint at: {path}")
                    break

            model = TRMModel(model_path=model_path, device=device)
            tokenizer = None  # TRM handles tokenization internally
            model_type = "trm"

            logger.info("✓ TRM model loaded successfully!")
            logger.info("TRM is a 7M parameter recursive model for reasoning tasks")

        except ImportError:
            raise RuntimeError(
                "TRM wrapper not found. Please run: python setup_trm.py"
            )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error loading TRM model: {error_msg}")

        if "not installed" in error_msg or "not found" in error_msg:
            logger.error("💡 To install TRM:")
            logger.error("   python setup_trm.py")

        raise RuntimeError(f"TRM model loading failed - {error_msg}")

def load_model():
    """Load selected model (Llama 3.1 or TRM) - GPU-ONLY MODE"""
    global model, tokenizer, device, model_type, selected_model

    try:
        # Load model configuration
        model_config = load_model_config()
        selected_model = model_config.get("selected_model", "llama3.1")

        logger.info(f"Selected model: {selected_model}")

        # Route to appropriate model loader
        if selected_model == "trm":
            load_trm_model()
            return
        else:
            # Default to Llama 3.1
            load_llama_model()

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error loading model: {error_msg}")
        raise RuntimeError(f"GPU-ONLY MODE: Model loading failed - {error_msg}")

def load_llama_model():
    """Load Llama 3.1 model - GPU-ONLY MODE"""
    global model, tokenizer, device, model_type

    try:
        # Get device configuration - GPU-ONLY
        device_config = get_device_config()
        device = device_config['device']  # Always 'cuda'
        
        logger.info(f"🚀 GPU-ONLY MODE: Using device: {device}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # First, try to find local Llama 3.1 8B model
        local_model_paths = [
            "./models/llama3.1-8b",     # Local Llama 3.1 8B path
            "/app/models/llama3.1-8b",  # Docker path
            "./models/llama3.1",        # Alternative local path
            "/app/models/llama3.1",     # Alternative Docker path
        ]
        
        model_path = None
        for path in local_model_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"Found local Llama 3.1 model at: {path}")
                break
        
        if model_path is not None:
            # Check if it's a GGUF model
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if config.get("format") == "gguf":
                        logger.info("Detected GGUF model format")
                        load_gguf_model(model_path)
                        return
                except Exception as e:
                    logger.warning(f"Could not read config.json: {e}")
            
            # Load as Hugging Face model
            logger.info("Loading local Llama 3.1 as Hugging Face model...")
            load_huggingface_model(model_path)
        else:
            # No local model found, try Ollama first, then Docker Hub
            logger.info("No local model found, trying Ollama...")
            try:
                load_ollama_model()
            except Exception as e:
                logger.warning(f"Ollama not available: {str(e)}")
                logger.info("Falling back to Docker Hub model...")
                load_dockerhub_model()
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error loading Llama 3.1 model: {error_msg}")
        
        if "Authentication required" in error_msg or "gated repo" in error_msg:
            logger.error("💡 Solutions:")
            logger.error("   • Use Docker: docker-compose up --build (recommended)")
            logger.error("   • Authenticate with Hugging Face: huggingface-cli login")
            logger.error("   • Set up local model: python setup.py")
        else:
            logger.error("GPU-ONLY MODE: Model loading failed. GPU is required.")
        
        raise RuntimeError(f"GPU-ONLY MODE: Llama 3.1 model loading failed - {error_msg}")

def load_gguf_model(model_path):
    """Load GGUF model using llama-cpp-python"""
    global model, tokenizer, device, model_type
    
    try:
        from llama_cpp import Llama
        
        # Find the .gguf file
        gguf_files = [f for f in os.listdir(model_path) if f.endswith('.gguf')]
        if not gguf_files:
            raise Exception("No .gguf file found in model directory")
        
        gguf_path = os.path.join(model_path, gguf_files[0])
        logger.info(f"Loading GGUF model: {gguf_path}")
        
        # Load the model with GPU support
        model = Llama(
            model_path=gguf_path,
            n_ctx=8192,  # Increased context window
            n_gpu_layers=-1 if device == "cuda" else 0,  # Use all GPU layers if CUDA available
            n_threads=4,  # Number of CPU threads
            verbose=False
        )
        
        model_type = "gguf"
        tokenizer = None  # GGUF models handle tokenization internally
        
        logger.info("GGUF model loaded successfully!")
        
    except ImportError:
        logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        raise
    except Exception as e:
        logger.error(f"Error loading GGUF model: {str(e)}")
        raise

def check_ollama_available():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def load_ollama_model():
    """Load Llama model using Ollama"""
    global model_type
    
    if not check_ollama_available():
        raise RuntimeError("Ollama is not running. Please start Ollama and ensure it's accessible at http://localhost:11434")
    
    # Check if llama3.1 model is available in Ollama
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            llama_models = [m for m in models if "llama" in m.get("name", "").lower()]
            
            if not llama_models:
                logger.warning("No Llama models found in Ollama. Please pull a Llama model first:")
                logger.warning("  ollama pull llama3.1:8b")
                raise RuntimeError("No Llama models available in Ollama")
            
            logger.info(f"Found Llama models in Ollama: {[m['name'] for m in llama_models]}")
            model_type = "ollama"
            logger.info("Ollama model loaded successfully!")
            return True
        else:
            raise RuntimeError(f"Failed to connect to Ollama: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to connect to Ollama: {str(e)}")

def load_dockerhub_model():
    """Load Llama 3.1 model from pre-downloaded Docker image or download from Hugging Face"""
    global model, tokenizer, device, model_type
    
    try:
        # Check if we're running in Docker or locally
        docker_model_path = "/app/models/llama3.1"
        local_model_path = "./models/llama3.1"
        
        if os.path.exists(docker_model_path):
            # Running in Docker - use pre-downloaded model
            model_path = docker_model_path
            logger.info(f"Loading pre-downloaded model from Docker: {model_path}")
            local_files_only = True
        elif os.path.exists(local_model_path):
            # Local development - use local model if available
            model_path = local_model_path
            logger.info(f"Loading local model: {model_path}")
            local_files_only = True
        else:
            # No local model - download Llama 3.1 8B from Hugging Face
            # ERRO ESPERADO: Requer autenticação no Hugging Face
            model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            logger.info(f"Attempting to download Llama 3.1 8B from Hugging Face: {model_name}")
            logger.warning("⚠️  ATENÇÃO: Este modelo requer autenticação no Hugging Face!")
            logger.warning("   Execute: huggingface-cli login")
            logger.warning("   E aceite os termos em: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
            model_path = model_name
            local_files_only = False
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Loading model...")

        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

        # GPU-ONLY model loading - Testing 8-bit quantization
        # 8-bit quantization: ~50% memory reduction, better quality than 4-bit
        logger.info("Testing 8-bit quantization for model loading...")

        try:
            # Try 8-bit quantization first
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=local_files_only
            )

            logger.info("✓ Model loaded with 8-bit quantization successfully!")

        except Exception as e8:
            logger.warning(f"8-bit quantization failed: {e8}")
            logger.info("Falling back to 4-bit quantization...")

            # Fallback to 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=local_files_only
            )

            logger.info("✓ Model loaded with 4-bit quantization (fallback)")

        # Log device mapping to verify GPU usage
        logger.info("Model loaded with quantization! Device mapping:")
        if hasattr(model, 'hf_device_map'):
            for name, dev in model.hf_device_map.items():
                logger.info(f"  {name}: {dev}")
        else:
            logger.info(f"  Model device: {next(model.parameters()).device}")

        # Log GPU memory usage
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        model_type = "huggingface"
        if local_files_only:
            logger.info("Model loaded successfully from local files!")
        else:
            logger.info("Model downloaded and loaded successfully from Hugging Face!")
        
    except Exception as e:
        error_msg = str(e)
        if "gated repo" in error_msg or "401 Client Error" in error_msg:
            logger.error("❌ Authentication required for Llama 3.1 model!")
            logger.error("🔑 To access this model, you need to:")
            logger.error("   1. Create a Hugging Face account at https://huggingface.co")
            logger.error("   2. Request access to the Llama 3.1 model")
            logger.error("   3. Install Hugging Face CLI: pip install huggingface_hub")
            logger.error("   4. Login: huggingface-cli login")
            logger.error("   5. Or use Docker with pre-downloaded model: docker-compose up --build")
            raise RuntimeError("Llama 3.1 model requires Hugging Face authentication. Please authenticate or use Docker.")
        else:
            logger.error(f"Error loading model: {error_msg}")
            raise

def load_huggingface_model(model_path):
    """Load Hugging Face model from local path with oLLM optimization"""
    global model, tokenizer, device, model_type

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Loading model...")

        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

        # GPU-ONLY model loading - Testing 8-bit quantization
        # 8-bit quantization: ~50% memory reduction, better quality than 4-bit
        logger.info("Testing 8-bit quantization for model loading...")

        try:
            # Try 8-bit quantization first
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )

            logger.info("✓ Model loaded with 8-bit quantization successfully!")

        except Exception as e8:
            logger.warning(f"8-bit quantization failed: {e8}")
            logger.info("Falling back to 4-bit quantization...")

            # Fallback to 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )

            logger.info("✓ Model loaded with 4-bit quantization (fallback)")

        # Log device mapping to verify GPU usage
        logger.info("Model loaded with quantization! Device mapping:")
        if hasattr(model, 'hf_device_map'):
            for name, dev in model.hf_device_map.items():
                logger.info(f"  {name}: {dev}")
        else:
            logger.info(f"  Model device: {next(model.parameters()).device}")

        # Log GPU memory usage
        logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        model_type = "huggingface"
        logger.info("Hugging Face model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading Hugging Face model: {str(e)}")
        raise

# DialoGPT fallback removed - using only Llama 3.1

# Ollama functionality removed - using only local Llama 3.1

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting up the application...")
    load_model()

    # Configure LlamaIndex with the loaded model
    if RAG_DB_AVAILABLE:
        configure_llamaindex_llm()

    yield
    # Shutdown
    logger.info("Shutting down the application...")

app = FastAPI(title="Llama 3.1 Chat API", version="1.0.0", lifespan=lifespan)

# Add CORS middleware
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main interface"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Llama 3.1 Chat</h1><p>Interface not found. Please check static/index.html</p>")

@app.get("/favicon.ico")
async def favicon():
    """Handle favicon requests to prevent 404 errors"""
    from fastapi.responses import Response
    return Response(status_code=204)  # No content response

# File upload endpoint
@app.post("/api/upload-file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and extract its content for context"""
    try:
        # Read file content
        file_content = await file.read()

        # Extract text from file
        extracted_text = extract_text_from_file(file.filename, file_content)

        # Store in global context (using filename as key for simplicity)
        # In production, use session IDs
        file_contexts[file.filename] = extracted_text

        # If RAG is available, automatically index the document
        if RAG_DB_AVAILABLE:
            try:
                doc_id, num_chunks = rag_llamaindex.index_document(
                    filename=file.filename,
                    content=extracted_text,
                    metadata={'upload_time': 'now'}
                )
                logger.info(f"Document indexed in RAG: {file.filename} (ID: {doc_id}, {num_chunks} chunks)")
            except Exception as e:
                logger.warning(f"Failed to index document in RAG: {e}")

        # Create preview (first 500 characters)
        preview = extracted_text[:500] + ("..." if len(extracted_text) > 500 else "")

        logger.info(f"File uploaded: {file.filename}, size: {len(file_content)} bytes, extracted: {len(extracted_text)} chars")

        return FileUploadResponse(
            filename=file.filename,
            content_preview=preview,
            status="success",
            message=f"Arquivo '{file.filename}' processado com sucesso! {len(extracted_text)} caracteres extraídos."
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_model(request: ChatRequest):
    """Chat with the selected model using LlamaIndex Query Engine"""
    try:
        # Check if any model type is loaded
        if model_type is None:
            # Return a helpful message when model is not available
            return ChatResponse(
                response="Desculpe, o modelo de IA não está disponível no momento. Verifique os logs do servidor para mais informações.",
                status="model_unavailable"
            )

        # OPÇÃO 2: Usar LlamaIndex Query Engine (RAG Completo)
        # Automatically handles retrieval + generation
        if RAG_DB_AVAILABLE and model_type == "huggingface":
            return chat_with_llamaindex_engine(request)

        # Fallback para outros model types
        # Handle TRM model
        if model_type == "trm":
            return chat_with_trm_model(request)
        # Handle Llama 3.1 model types
        elif model_type == "gguf":
            return chat_with_gguf_model(request)
        elif model_type == "ollama":
            return chat_with_ollama_model(request)
        else:
            return chat_with_huggingface_model(request)

    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return ChatResponse(
            response=f"I encountered an error while processing your request: {str(e)}",
            status="error"
        )

# TensorFlow functionality removed - using only Llama3

# Ollama chat function removed - using only local Llama 3.1

def chat_with_trm_model(request: ChatRequest):
    """Chat with TRM (Tiny Recursive Models) model"""
    try:
        # Build context with conversation history
        context = ""

        # Add file context if provided
        if request.file_context and request.file_context in file_contexts:
            file_content = file_contexts[request.file_context]
            context += f"Context from file '{request.file_context}':\n{file_content[:2000]}\n\n"

        # Add conversation history
        for msg in request.history:
            if msg.role == "user":
                context += f"User: {msg.content}\n"
            elif msg.role == "assistant":
                context += f"Assistant: {msg.content}\n"

        # Add current message
        full_prompt = context + f"User: {request.message}\nAssistant:"

        # Generate response using TRM recursive approach
        # TRM is designed for reasoning tasks, so we adjust parameters
        response_text = model.generate(
            prompt=full_prompt,
            max_iterations=10,  # TRM uses recursive refinement
            temperature=0.7
        )

        return ChatResponse(response=response_text, status="success")

    except Exception as e:
        logger.error(f"Error in TRM chat: {str(e)}")
        return ChatResponse(
            response=f"I encountered an error while processing your request: {str(e)}",
            status="error"
        )

def chat_with_gguf_model(request: ChatRequest):
    """Chat with GGUF model"""
    try:
        # Build prompt with conversation history
        prompt = "<|begin_of_text|>"

        # Add file context if provided
        if request.file_context and request.file_context in file_contexts:
            file_content = file_contexts[request.file_context]
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\nContexto do arquivo '{request.file_context}':\n{file_content[:2000]}<|eot_id|>"

        # Add conversation history
        for msg in request.history:
            if msg.role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"

        # Add current user message
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{request.message}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # Get generation parameters from environment
        max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', 1024))
        temperature = float(os.getenv('TEMPERATURE', 0.7))
        top_p = float(os.getenv('TOP_P', 0.9))

        # Generate response using GGUF model
        response = model(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )

        # Extract the response text
        response_text = response['choices'][0]['text'].strip()

        return ChatResponse(response=response_text, status="success")

    except Exception as e:
        logger.error(f"Error in GGUF chat: {str(e)}")
        return ChatResponse(
            response=f"I encountered an error while processing your request: {str(e)}",
            status="error"
        )

def chat_with_ollama_model(request: ChatRequest):
    """Chat with Ollama Llama 3.1 8B model"""
    try:
        # Build the full context with history
        context = ""

        # Add file context if provided
        if request.file_context and request.file_context in file_contexts:
            file_content = file_contexts[request.file_context]
            context += f"Contexto do arquivo '{request.file_context}':\n{file_content[:2000]}\n\n"

        for msg in request.history:
            if msg.role == "user":
                context += f"Usuário: {msg.content}\n"
            elif msg.role == "assistant":
                context += f"Assistente: {msg.content}\n"

        # Add current message
        full_prompt = context + f"Usuário: {request.message}\nAssistente:"

        # Prepare the request for Ollama API
        ollama_request = {
            "model": "llama3.1:8b",
            "prompt": full_prompt,
            "stream": False
        }

        # Send request to Ollama
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=ollama_request,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return ChatResponse(
                response=result.get("response", "Sem resposta do Ollama"),
                status="success"
            )
        else:
            return ChatResponse(
                response=f"Erro: API Ollama retornou status {response.status_code}",
                status="error"
            )

    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama: {str(e)}")
        return ChatResponse(
            response=f"Erro ao conectar com Ollama: {str(e)}",
            status="error"
        )
    except Exception as e:
        logger.error(f"Error in Ollama chat: {str(e)}")
        return ChatResponse(
            response=f"Erro: {str(e)}",
            status="error"
        )

def clean_response(text: str) -> str:
    """
    Remove redundância e limpa respostas excessivamente longas.
    Focado em extrair APENAS a resposta direta à pergunta.
    """
    import re

    # 1. Limpar números isolados e lixo comum
    text = re.sub(r'^\s*[\d.]+\s*$', '', text, flags=re.MULTILINE)

    # 2. Remover linhas com perguntas "alucinadas" (contêm ponto de interrogação ou começam com palavras interrogativas)
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Ignorar linhas que parecem perguntas (contêm ? ou começam com palavras interrogativas)
        is_question = ('?' in line_stripped or
                      re.match(r'^(Qual|Quais|Como|Quando|Onde|Por que|O que|Que tipo|Em que)\b', line_stripped, re.IGNORECASE))
        if line_stripped and not is_question:
            filtered_lines.append(line)

    text = '\n'.join(filtered_lines)

    # 3. Pegar apenas a primeira sentença com conteúdo substantivo
    # Dividir em sentenças
    sentences = re.split(r'[.!?]\s+', text.strip())

    # Encontrar a primeira sentença que contenha informação relevante
    # (não seja apenas um número ou muito curta)
    first_meaningful_sentence = None
    for sentence in sentences:
        sentence_clean = sentence.strip()
        # Deve ter pelo menos 20 caracteres e não ser só números
        if len(sentence_clean) >= 20 and not re.match(r'^[\d\s,.]+$', sentence_clean):
            first_meaningful_sentence = sentence_clean
            break

    if first_meaningful_sentence:
        result = first_meaningful_sentence
        # Adicionar ponto final se não houver
        if not result.endswith(('.', '!', '?')):
            result += '.'
    else:
        # Fallback: pegar as primeiras 200 caracteres do texto original
        result = text[:200].strip()
        if result and not result.endswith(('.', '!', '?')):
            result += '...'

    # 4. Remover padrões conhecidos de redundância
    result = re.sub(r'\s*Explanation:.*', '', result, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(r'\s*Portanto,.*', '', result, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(r'\s*Em relação.*', '', result, flags=re.DOTALL | re.IGNORECASE)
    result = re.sub(r'\s*Não existe informações.*', '', result, flags=re.DOTALL | re.IGNORECASE)

    # 5. Limitar tamanho máximo (para segurança)
    if len(result) > 300:
        result = result[:300].rsplit('.', 1)[0] + '.'

    return result.strip()


def chat_with_llamaindex_engine(request: ChatRequest):
    """
    Chat using LlamaIndex Query Engine (Opção 2: Query Engine Completo)
    Usa LlamaIndex para retrieval automático + geração de resposta
    """
    try:
        # Configure LlamaIndex LLM if needed
        configure_llamaindex_llm()

        # Get RAG configuration from environment
        rag_top_k = int(os.getenv('RAG_TOP_K', 3))
        rag_min_similarity = float(os.getenv('RAG_MIN_SIMILARITY', 0.6))

        # Create query engine from LlamaIndex
        query_engine = rag_llamaindex.create_query_engine(
            top_k=rag_top_k,
            similarity_threshold=rag_min_similarity,
            response_mode="compact"  # compact, tree_summarize, simple_summarize
        )

        # Build query with context from history
        # LlamaIndex query engines don't natively support conversation history,
        # so we'll prepend it to the query
        context_query = ""

        # Add conversation history
        if request.history:
            context_query += "Histórico da conversa:\n"
            for msg in request.history[-3:]:  # Últimas 3 mensagens
                context_query += f"{msg.role}: {msg.content}\n"
            context_query += "\n"

        # Add current message
        context_query += f"Pergunta atual: {request.message}"

        # Query the engine (automatically retrieves + generates)
        logger.info(f"🦙 LlamaIndex Query Engine processing: {request.message[:50]}...")
        response = query_engine.query(context_query)

        # Extract response text
        response_text = str(response.response) if hasattr(response, 'response') else str(response)

        # Post-process: Remove redundancy and excessive explanations
        response_text = clean_response(response_text)

        logger.info(f"✓ LlamaIndex generated response ({len(response_text)} chars)")

        return ChatResponse(
            response=response_text,
            status="success"
        )

    except Exception as e:
        logger.error(f"Error in LlamaIndex Query Engine: {str(e)}")
        # Fallback to standard huggingface chat
        logger.info("Falling back to standard Hugging Face chat...")
        return chat_with_huggingface_model(request)


def chat_with_huggingface_model(request: ChatRequest):
    """Chat with Hugging Face Llama 3.1 8B model"""
    try:
        # Build prompt with conversation history using Llama 3.1 format
        prompt = "<|begin_of_text|>"

        # Add RAG context if available
        rag_context = ""
        if RAG_DB_AVAILABLE:
            try:
                # Get RAG configuration from environment
                rag_top_k = int(os.getenv('RAG_TOP_K', 3))
                rag_min_similarity = float(os.getenv('RAG_MIN_SIMILARITY', 0.6))

                # Search for relevant chunks using the user's message
                similar_chunks = rag_llamaindex.search_similar_chunks(
                    query=request.message,
                    top_k=rag_top_k,
                    min_similarity=rag_min_similarity
                )

                if similar_chunks:
                    rag_context = "\n\n=== Contexto Relevante de Documentos ===\n"
                    for chunk in similar_chunks:
                        rag_context += f"\n[Fonte: {chunk['filename']}, Similaridade: {chunk['similarity']:.2f}]\n"
                        rag_context += f"{chunk['chunk_text'][:500]}\n"

                    logger.info(f"RAG: Found {len(similar_chunks)} relevant chunks for query")
            except Exception as e:
                logger.warning(f"RAG search failed, continuing without context: {e}")

        # Add system context (RAG + file context)
        system_context = ""
        if rag_context:
            system_context += rag_context

        # Add file context if provided
        if request.file_context and request.file_context in file_contexts:
            file_content = file_contexts[request.file_context]
            system_context += f"\n\n=== Contexto do Arquivo '{request.file_context}' ===\n{file_content[:2000]}"

        if system_context:
            prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_context}<|eot_id|>"

        # Add conversation history
        for msg in request.history:
            if msg.role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
            elif msg.role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"

        # Add current user message
        prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{request.message}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

        # Get generation parameters from environment
        max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', 1024))
        temperature = float(os.getenv('TEMPERATURE', 0.7))
        top_p = float(os.getenv('TOP_P', 0.9))

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else None
            )

        # Decode ONLY the new tokens (not including the input prompt)
        # outputs[0] contains: [input_ids + generated_ids]
        # We want only the generated part
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up common tokens that might appear
        response = response.replace("<|eot_id|>", "").strip()
        response = response.replace("<|end_of_text|>", "").strip()
        response = response.replace("Output:", "").strip()
        response = response.replace("Assistant:", "").strip()

        return ChatResponse(response=response, status="success")

    except Exception as e:
        logger.error(f"Error in Hugging Face chat: {str(e)}")
        return ChatResponse(
            response=f"I encountered an error while processing your request: {str(e)}",
            status="error"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint - GPU-ONLY MODE"""
    # Get device configuration - GPU-ONLY
    try:
        get_device_config()  # Just validate GPU is available
    except Exception as e:
        return {
            "status": "error",
            "error": f"GPU-ONLY MODE: {str(e)}",
            "cuda_available": False
        }

    # GPU information (always available in GPU-ONLY mode)
    gpu_info = {
        "device_name": torch.cuda.get_device_name(0),
        "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB",
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GB",
        "memory_cached": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f} GB"
    }

    health_info = {
        "status": "healthy",
        "mode": "GPU_ONLY",
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": device,
        "cuda_available": True,  # Always true in GPU-ONLY mode
        "gpu_info": gpu_info,
        "postgres_ai_available": POSTGRES_AI_AVAILABLE,
        "ollama_available": check_ollama_available()
    }

    # TensorFlow QA system removed - using only Llama3

    return health_info

@app.get("/api/gpu-stats")
async def gpu_stats():
    """Detailed GPU statistics endpoint"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    stats = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "memory": {
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3,
            "utilization_percent": (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100
        },
        "model_info": {
            "loaded": model is not None,
            "type": model_type,
            "device": str(device)
        }
    }

    # Add device mapping if model is loaded
    if model is not None and hasattr(model, 'hf_device_map'):
        stats["model_info"]["device_map"] = {str(k): str(v) for k, v in model.hf_device_map.items()}
    elif model is not None:
        try:
            stats["model_info"]["model_device"] = str(next(model.parameters()).device)
        except:
            stats["model_info"]["model_device"] = "unknown"

    return stats

# RAG Endpoints
@app.post("/api/rag/index", response_model=RAGIndexResponse)
async def rag_index_document(request: RAGIndexRequest):
    """Index a document in the RAG database"""
    if not RAG_DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG database não está disponível")

    try:
        doc_id, num_chunks = rag_llamaindex.index_document(
            filename=request.filename,
            content=request.content,
            metadata=request.metadata
        )

        return RAGIndexResponse(
            document_id=doc_id,
            num_chunks=num_chunks,
            status="success",
            message=f"Documento indexado com sucesso! ID: {doc_id}, {num_chunks} chunks criados."
        )

    except Exception as e:
        logger.error(f"Error indexing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao indexar documento: {str(e)}")

@app.post("/api/rag/search", response_model=RAGSearchResponse)
async def rag_search(request: RAGSearchRequest):
    """Search for similar documents using RAG"""
    if not RAG_DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG database não está disponível")

    try:
        results = rag_llamaindex.search_similar_chunks(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )

        return RAGSearchResponse(
            results=results,
            status="success"
        )

    except Exception as e:
        logger.error(f"Error searching RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao buscar no RAG: {str(e)}")

@app.get("/api/rag/stats", response_model=RAGStatsResponse)
async def rag_stats():
    """Get RAG database statistics"""
    if not RAG_DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG database não está disponível")

    try:
        stats = rag_llamaindex.get_database_stats()

        return RAGStatsResponse(
            stats=stats,
            status="success"
        )

    except Exception as e:
        logger.error(f"Error getting RAG stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter estatísticas do RAG: {str(e)}")

@app.get("/api/rag/documents")
async def rag_list_documents(limit: int = 100, offset: int = 0):
    """List all indexed documents"""
    if not RAG_DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG database não está disponível")

    try:
        documents = rag_llamaindex.list_documents(limit=limit, offset=offset)

        return {
            "documents": documents,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao listar documentos: {str(e)}")

@app.delete("/api/rag/documents/{document_id}")
async def rag_delete_document(document_id: int):
    """Delete a document from RAG database"""
    if not RAG_DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG database não está disponível")

    try:
        deleted = rag_llamaindex.delete_document(document_id)

        if deleted:
            return {"status": "success", "message": f"Documento {document_id} deletado com sucesso"}
        else:
            raise HTTPException(status_code=404, detail=f"Documento {document_id} não encontrado")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao deletar documento: {str(e)}")

# File testing endpoint removed

# Cache reset endpoint removed - no file operations

# File deletion endpoint removed

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Allow custom port via command line argument
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.warning(f"Invalid port '{sys.argv[1]}', using default port 8000")
    
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
