"""
RAG Module usando LlamaIndex com PostgreSQL + pgvector
Substitui o rag_db.py com uma implementação mais simples e poderosa
"""
import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Database connection configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5433)),
    'database': os.getenv('DB_NAME', 'vetorial_bd'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'agro123')
}

# RAG Configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
DEFAULT_CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))

logger.info(f"RAG LlamaIndex Configuration:")
logger.info(f"  - Database: {DB_CONFIG['user']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
logger.info(f"  - Embedding Model: {EMBEDDING_MODEL_NAME}")
logger.info(f"  - Chunk Size: {DEFAULT_CHUNK_SIZE} chars")
logger.info(f"  - Chunk Overlap: {DEFAULT_CHUNK_OVERLAP} chars")

# Global variables
_vector_store = None
_index = None
_embed_model = None


def get_embedding_model():
    """Get or initialize the embedding model"""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_MODEL_NAME,
            device='cuda'  # Use GPU
        )
        logger.info("Embedding model loaded successfully")
    return _embed_model


def get_vector_store():
    """Get or initialize the vector store"""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing PGVector store...")

        # Create connection string
        connection_string = (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

        _vector_store = PGVectorStore.from_params(
            database=DB_CONFIG['database'],
            host=DB_CONFIG['host'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port'],
            user=DB_CONFIG['user'],
            table_name="llamaindex_vectors",  # Nova tabela para LlamaIndex
            embed_dim=384,  # all-MiniLM-L6-v2 dimensions
            hybrid_search=False,  # Pode ativar hybrid search no futuro
            text_search_config="portuguese"  # Configuração para português
        )

        logger.info("PGVector store initialized successfully")

    return _vector_store


def get_index():
    """Get or initialize the vector store index"""
    global _index
    if _index is None:
        logger.info("Initializing VectorStoreIndex...")

        # Configure global settings
        Settings.embed_model = get_embedding_model()
        Settings.chunk_size = DEFAULT_CHUNK_SIZE
        Settings.chunk_overlap = DEFAULT_CHUNK_OVERLAP

        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=get_vector_store()
        )

        # Try to load existing index or create new one
        try:
            _index = VectorStoreIndex.from_vector_store(
                vector_store=get_vector_store(),
                storage_context=storage_context
            )
            logger.info("Loaded existing VectorStoreIndex")
        except Exception as e:
            logger.info(f"Creating new VectorStoreIndex: {e}")
            _index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context
            )

        logger.info("VectorStoreIndex initialized successfully")

    return _index


def index_document(
    filename: str,
    content: str,
    metadata: Optional[Dict] = None
) -> tuple[str, int]:
    """
    Index a document using LlamaIndex

    Args:
        filename: Name of the document
        content: Full text content
        metadata: Optional metadata dictionary

    Returns:
        Tuple of (document_id, number_of_nodes)
    """
    try:
        logger.info(f"Indexing document: {filename}")

        # Prepare metadata
        doc_metadata = metadata or {}
        doc_metadata['filename'] = filename

        # Create LlamaIndex Document
        document = Document(
            text=content,
            metadata=doc_metadata
        )

        # Get index
        index = get_index()

        # Insert document (automatically chunks and embeds)
        index.insert(document)

        # Count nodes (chunks) - estimativa baseada no tamanho do texto
        estimated_nodes = max(1, len(content) // DEFAULT_CHUNK_SIZE)

        logger.info(f"Document indexed successfully: {filename} (~{estimated_nodes} nodes)")

        return (filename, estimated_nodes)

    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        raise


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.5
) -> List[Dict]:
    """
    Search for similar documents/chunks

    Args:
        query: Search query text
        top_k: Number of top results to return
        min_similarity: Minimum similarity threshold (0-1)

    Returns:
        List of dictionaries with chunk information and similarity scores
    """
    try:
        logger.info(f"Searching for: {query[:50]}...")

        # Get index
        index = get_index()

        # Use retriever directly (não precisa de LLM)
        retriever = index.as_retriever(
            similarity_top_k=top_k
        )

        # Execute retrieval
        nodes = retriever.retrieve(query)

        # Process results
        results = []
        for node in nodes:
            # Calcular similaridade (score do LlamaIndex já é normalizado)
            similarity = node.score if node.score is not None else 0.0

            # Filtrar por similaridade mínima
            if similarity >= min_similarity:
                results.append({
                    'id': node.node_id,
                    'chunk_text': node.text,
                    'chunk_index': node.metadata.get('chunk_index', 0),
                    'filename': node.metadata.get('filename', 'unknown'),
                    'metadata': node.metadata,
                    'similarity': similarity
                })

        logger.info(f"Found {len(results)} similar chunks")
        return results

    except Exception as e:
        logger.error(f"Failed to search similar chunks: {e}")
        raise


def get_retriever(top_k: int = 3, similarity_threshold: float = 0.6):
    """
    Get a retriever object for use in query engines

    Args:
        top_k: Number of top results
        similarity_threshold: Minimum similarity

    Returns:
        VectorIndexRetriever
    """
    index = get_index()
    return index.as_retriever(
        similarity_top_k=top_k,
        similarity_cutoff=similarity_threshold
    )


def create_query_engine(
    top_k: int = 3,
    similarity_threshold: float = 0.6,
    response_mode: str = "compact"
):
    """
    Create a query engine for RAG

    Args:
        top_k: Number of chunks to retrieve
        similarity_threshold: Minimum similarity
        response_mode: "compact", "tree_summarize", "simple_summarize", etc.

    Returns:
        QueryEngine
    """
    index = get_index()
    return index.as_query_engine(
        similarity_top_k=top_k,
        response_mode=response_mode,
        similarity_cutoff=similarity_threshold
    )


def get_database_stats() -> Dict:
    """
    Get database statistics

    Returns:
        Dictionary with statistics
    """
    try:
        # Com LlamaIndex, estatísticas detalhadas requerem queries SQL diretas
        # Vamos retornar informações básicas

        stats = {
            'embedding_model': EMBEDDING_MODEL_NAME,
            'embedding_dimensions': 384,
            'chunk_size': DEFAULT_CHUNK_SIZE,
            'chunk_overlap': DEFAULT_CHUNK_OVERLAP,
            'table_name': 'llamaindex_vectors',
            'status': 'operational'
        }

        # Tentar obter contagem de documentos
        try:
            from sqlalchemy import create_engine, text

            connection_string = (
                f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
                f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            )

            engine = create_engine(connection_string)
            with engine.connect() as conn:
                # Contar documentos na tabela do LlamaIndex
                result = conn.execute(text(
                    "SELECT COUNT(*) FROM llamaindex_vectors"
                ))
                count = result.scalar()
                stats['total_chunks'] = count

        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            stats['total_chunks'] = 'unknown'

        return stats

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            'error': str(e),
            'status': 'error'
        }


def delete_all_documents():
    """
    Delete all documents from the index (use with caution!)
    """
    try:
        logger.warning("Deleting all documents from index...")

        # Recreate index from scratch
        global _index
        storage_context = StorageContext.from_defaults(
            vector_store=get_vector_store()
        )

        _index = VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context
        )

        logger.info("All documents deleted successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to delete documents: {e}")
        raise


# Compatibility functions para manter compatibilidade com código antigo
def list_documents(limit: int = 100, offset: int = 0) -> List[Dict]:
    """
    List documents (compatibility function)
    LlamaIndex não tem essa funcionalidade nativa, então retornamos info básica
    """
    logger.warning("list_documents is not fully supported in LlamaIndex. Use get_database_stats() instead.")
    return []


def delete_document(document_id: int) -> bool:
    """
    Delete a specific document (compatibility function)
    LlamaIndex não suporta deleção de documentos individuais facilmente
    """
    logger.warning("delete_document is not supported in LlamaIndex. Use delete_all_documents() to clear all.")
    return False


def get_document_by_id(document_id: int) -> Optional[Dict]:
    """
    Get document by ID (compatibility function)
    """
    logger.warning("get_document_by_id is not supported in LlamaIndex.")
    return None


# Initialize on module import
try:
    logger.info("Initializing LlamaIndex RAG module...")

    # Disable default LLM (we'll configure it later in main.py if needed)
    from llama_index.core.llms import MockLLM
    Settings.llm = MockLLM()  # Use mock LLM for retrieval-only operations

    get_embedding_model()
    get_vector_store()
    get_index()
    logger.info("✅ LlamaIndex RAG module initialized successfully!")
except Exception as e:
    logger.error(f"❌ Failed to initialize LlamaIndex RAG module: {e}")
    logger.error("Please ensure PostgreSQL with pgvector is running and accessible")
