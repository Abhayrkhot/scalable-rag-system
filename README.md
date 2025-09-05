# Scalable RAG System

A high-performance Retrieval-Augmented Generation (RAG) system designed to handle millions of documents with enterprise-grade scalability and monitoring.

## Features

- **Massive Scale**: Process and query millions of documents efficiently
- **Multiple Vector DBs**: Support for ChromaDB (default) and Pinecone
- **Flexible Embeddings**: OpenAI embeddings with sentence-transformers fallback
- **Async Processing**: High-performance async/await architecture
- **Batch Processing**: Efficient batch processing for large document ingestion
- **Monitoring**: Built-in Prometheus metrics and structured logging
- **Cloud Ready**: Docker and Kubernetes deployment configurations
- **API Authentication**: Secure API key-based authentication
- **Multiple File Formats**: PDF, Markdown, and TXT support

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd scalable-rag-system
cp .env.example .env
# Edit .env with your configuration
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Edit `.env` file:

```env
API_KEY=your-secure-api-key-here
OPENAI_API_KEY=sk-your-openai-key-here
VECTOR_DB_PROVIDER=chroma  # or pinecone
EMBEDDING_MODEL=text-embedding-3-large
```

### 4. Run with Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f rag-api
```

### 5. Run Locally

```bash
# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start monitoring
python -c "from app.utils.monitoring import start_monitoring; start_monitoring()"
```

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Ingest Documents

```bash
curl -X POST "http://localhost:8000/ingest/" \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["sample_docs/document1.pdf", "sample_docs/document2.md"],
    "collection_name": "my_documents",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "batch_size": 100
  }'
```

### Query Documents

```bash
curl -X POST "http://localhost:8000/query/" \
  -H "x-api-key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is machine learning?",
    "collection_name": "my_documents",
    "top_k": 5,
    "rerank": true,
    "include_metadata": true
  }'
```

### Get Collection Info

```bash
curl -X GET "http://localhost:8000/ingest/collections/my_documents" \
  -H "x-api-key: your-api-key"
```

## Configuration

### Vector Database Options

#### ChromaDB (Default)
```env
VECTOR_DB_PROVIDER=chroma
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
```

#### Pinecone
```env
VECTOR_DB_PROVIDER=pinecone
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=rag-documents
```

### Embedding Models

#### OpenAI Embeddings (Recommended)
```env
EMBEDDING_MODEL=text-embedding-3-large
# or
EMBEDDING_MODEL=text-embedding-3-small
```

#### Sentence Transformers (Free)
```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:8001/metrics`

Key metrics:
- `rag_requests_total` - Total API requests
- `rag_request_duration_seconds` - Request duration
- `rag_documents_ingested_total` - Documents ingested
- `rag_chunks_created_total` - Chunks created
- `rag_query_duration_seconds` - Query processing time

### Logging

Structured JSON logging with configurable levels:

```env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## Performance Tuning

### For Millions of Documents

1. **Use Pinecone** for production scale:
   ```env
   VECTOR_DB_PROVIDER=pinecone
   ```

2. **Optimize batch sizes**:
   ```env
   EMBEDDING_BATCH_SIZE=1000
   ```

3. **Increase concurrency**:
   ```env
   MAX_CONCURRENT_INGESTIONS=20
   ```

4. **Use async processing**:
   - All operations are async by default
   - Use background tasks for large ingestions

### Memory Optimization

1. **Chunk size tuning**:
   ```env
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

2. **Batch processing**:
   - Process documents in batches
   - Use streaming for very large files

## Deployment

### Docker

```bash
# Build image
docker build -t scalable-rag .

# Run container
docker run -p 8000:8000 -p 8001:8001 \
  -e API_KEY=your-key \
  -e OPENAI_API_KEY=your-openai-key \
  scalable-rag
```

### Kubernetes

See `k8s/` directory for Kubernetes manifests.

### Cloud Deployment

#### AWS ECS/Fargate
- Use the provided Dockerfile
- Configure environment variables
- Set up RDS for PostgreSQL
- Use ElastiCache for Redis

#### Google Cloud Run
- Deploy as containerized service
- Use Cloud SQL for PostgreSQL
- Use Memorystore for Redis

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_ingestion.py
```

## Development

### Project Structure

```
scalable-rag-system/
├── app/
│   ├── core/           # Core business logic
│   ├── routers/        # FastAPI routers
│   ├── services/       # Service layer
│   ├── models/         # Pydantic models
│   └── utils/          # Utilities
├── tests/              # Test files
├── sample_docs/        # Sample documents
├── docker-compose.yml  # Docker Compose config
├── Dockerfile         # Docker image
└── requirements.txt   # Python dependencies
```

### Adding New Features

1. **New file types**: Extend `DocumentProcessor`
2. **New vector DBs**: Extend `VectorStoreManager`
3. **New endpoints**: Add to `routers/`
4. **New metrics**: Add to `utils/monitoring.py`

## Troubleshooting

### Common Issues

1. **OpenAI API errors**: Check API key and rate limits
2. **Vector DB connection**: Verify configuration
3. **Memory issues**: Reduce batch sizes
4. **Slow ingestion**: Use Pinecone for large scale

### Logs

```bash
# View application logs
docker-compose logs -f rag-api

# View specific service logs
docker-compose logs -f postgres
docker-compose logs -f redis
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the logs for errors
