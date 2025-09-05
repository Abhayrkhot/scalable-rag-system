import os
from typing import Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    api_key: str = Field(..., env="API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Vector Database Configuration
    vector_db_provider: str = Field("chroma", env="VECTOR_DB_PROVIDER")
    chroma_persist_directory: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    pinecone_api_key: Optional[str] = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: Optional[str] = Field(None, env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("rag-documents", env="PINECONE_INDEX_NAME")
    
    # Embedding Configuration
    embedding_model: str = Field("text-embedding-3-large", env="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(100, env="EMBEDDING_BATCH_SIZE")
    embedding_dimension: int = Field(3072, env="EMBEDDING_DIMENSION")
    
    # Database Configuration
    database_url: str = Field("sqlite:///./rag_system.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # Celery Configuration
    celery_broker_url: str = Field("redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field("redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Application Configuration
    max_file_size_mb: int = Field(100, env="MAX_FILE_SIZE_MB")
    max_concurrent_ingestions: int = Field(10, env="MAX_CONCURRENT_INGESTIONS")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    max_query_results: int = Field(20, env="MAX_QUERY_RESULTS")
    
    # Monitoring
    prometheus_port: int = Field(8001, env="PROMETHEUS_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Security Configuration
    max_request_size_mb: int = Field(10, env="MAX_REQUEST_SIZE_MB")
    rate_limit_per_minute: int = Field(100, env="RATE_LIMIT_PER_MINUTE")
    allowed_file_types: str = Field("pdf,txt,md,markdown", env="ALLOWED_FILE_TYPES")
    
    # Hybrid Search Configuration
    elasticsearch_host: str = Field("localhost", env="ELASTICSEARCH_HOST")
    elasticsearch_port: int = Field(9200, env="ELASTICSEARCH_PORT")
    enable_hybrid_search: bool = Field(True, env="ENABLE_HYBRID_SEARCH")
    
    # Reranking Configuration
    reranker_type: str = Field("cross_encoder", env="RERANKER_TYPE")  # cross_encoder, bge_reranker, cohere
    reranker_cache_ttl: int = Field(3600, env="RERANKER_CACHE_TTL")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    
    # Guardrails Configuration
    max_tokens: int = Field(4000, env="MAX_TOKENS")
    max_context_tokens: int = Field(8000, env="MAX_CONTEXT_TOKENS")
    require_citations: bool = Field(True, env="REQUIRE_CITATIONS")
    forbid_unverifiable: bool = Field(True, env="FORBID_UNVERIFIABLE")
    min_confidence_threshold: float = Field(0.3, env="MIN_CONFIDENCE_THRESHOLD")
    max_sources: int = Field(10, env="MAX_SOURCES")
    
    # Evaluation Configuration
    enable_evaluation: bool = Field(True, env="ENABLE_EVALUATION")
    evaluation_dataset_path: str = Field("./evaluation_data", env="EVALUATION_DATASET_PATH")
    
    # Performance Configuration
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    enable_streaming: bool = Field(True, env="ENABLE_STREAMING")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
