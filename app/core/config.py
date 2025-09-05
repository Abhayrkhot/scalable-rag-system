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
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
