.PHONY: help install test eval clean build run dev prod docker-compose-up docker-compose-down

# Default target
help:
	@echo "Available targets:"
	@echo "  install          - Install dependencies"
	@echo "  test             - Run tests"
	@echo "  eval             - Run evaluation suite"
	@echo "  clean            - Clean up temporary files"
	@echo "  build            - Build Docker image"
	@echo "  run              - Run the application"
	@echo "  dev              - Run in development mode"
	@echo "  prod             - Run in production mode"
	@echo "  docker-compose-up - Start all services with Docker Compose"
	@echo "  docker-compose-down - Stop all services"
	@echo "  lint             - Run linting"
	@echo "  format           - Format code"
	@echo "  type-check       - Run type checking"

# Installation
install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

# Evaluation
eval: reports
	python run_evaluation.py
	@echo "Evaluation completed. Check reports/ directory for results."

reports:
	mkdir -p reports

# Code quality
lint:
	ruff check app/
	black --check app/
	mypy app/

format:
	ruff check --fix app/
	black app/

type-check:
	mypy app/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf reports/
	rm -rf data/
	rm -rf logs/

# Docker
build:
	docker build -t rag-system:latest .

run:
	docker run -p 8000:8000 -p 8001:8001 rag-system:latest

# Development
dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker Compose
docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Database operations
migrate:
	python scripts/migrate_embeddings.py --collection $(COLLECTION) --new-model $(MODEL) --dry-run

migrate-force:
	python scripts/migrate_embeddings.py --collection $(COLLECTION) --new-model $(MODEL) --force

# Monitoring
metrics:
	@echo "Prometheus metrics available at http://localhost:8001/metrics"
	@echo "Grafana dashboard available at http://localhost:3000"

# Health checks
health:
	curl http://localhost:8000/health/

ready:
	curl http://localhost:8000/health/ready

status:
	curl http://localhost:8000/health/status

# Performance testing
load-test:
	@echo "Running load test..."
	# Add load testing commands here

# Security scanning
security:
	bandit -r app/
	safety check

# Documentation
docs:
	@echo "API documentation available at http://localhost:8000/docs"
	@echo "ReDoc documentation available at http://localhost:8000/redoc"

# Full CI pipeline
ci: install lint type-check test security eval
	@echo "CI pipeline completed successfully"

# Development setup
setup-dev: install
	@echo "Setting up development environment..."
	mkdir -p data/chroma_db
	mkdir -p logs
	mkdir -p reports
	@echo "Development environment ready!"

# Production setup
setup-prod: install
	@echo "Setting up production environment..."
	mkdir -p data/chroma_db
	mkdir -p logs
	mkdir -p reports
	@echo "Production environment ready!"

# Backup
backup:
	@echo "Creating backup..."
	tar -czf backup-$(shell date +%Y%m%d-%H%M%S).tar.gz data/ logs/ reports/
	@echo "Backup created"

# Restore
restore:
	@echo "Restoring from backup..."
	tar -xzf $(BACKUP_FILE)
	@echo "Restore completed"

# Update dependencies
update-deps:
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt

# Generate requirements
generate-requirements:
	pip freeze > requirements-generated.txt

# Database operations
db-reset:
	@echo "Resetting database..."
	rm -rf data/chroma_db/*
	@echo "Database reset completed"

# Cache operations
cache-clear:
	@echo "Clearing cache..."
	curl -X POST http://localhost:8000/admin/cache/clear
	@echo "Cache cleared"

# Collection operations
list-collections:
	@echo "Listing collections..."
	curl http://localhost:8000/query/collections

# Evaluation with specific dataset
eval-dataset:
	python run_evaluation.py --dataset $(DATASET)

# Performance benchmark
benchmark:
	@echo "Running performance benchmark..."
	python scripts/benchmark.py

# Memory profiling
profile:
	@echo "Running memory profiling..."
	python -m memory_profiler app/main.py

# CPU profiling
profile-cpu:
	@echo "Running CPU profiling..."
	python -m cProfile -o profile.prof app/main.py

# Generate reports
reports: reports
	@echo "Generating reports..."
	python scripts/generate_reports.py
	@echo "Reports generated in reports/ directory"

# Full system test
system-test: docker-compose-up
	@echo "Running full system test..."
	sleep 30  # Wait for services to start
	make test
	make eval
	make docker-compose-down
	@echo "System test completed"

# Quick start
quick-start: setup-dev
	@echo "Starting RAG system..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "RAG system started at http://localhost:8000"
	@echo "API docs at http://localhost:8000/docs"
	@echo "Metrics at http://localhost:8001/metrics"

# Stop quick start
stop:
	pkill -f "uvicorn app.main:app"

# Show logs
logs:
	tail -f logs/app.log

# Show system status
status-all:
	@echo "=== System Status ==="
	@echo "Application:"
	@make health
	@echo ""
	@echo "Metrics:"
	@make metrics
	@echo ""
	@echo "Collections:"
	@make list-collections

# Help for specific targets
help-eval:
	@echo "Evaluation targets:"
	@echo "  eval             - Run full evaluation suite"
	@echo "  eval-dataset     - Run evaluation on specific dataset"
	@echo "  benchmark        - Run performance benchmark"
	@echo "  reports          - Generate evaluation reports"

help-docker:
	@echo "Docker targets:"
	@echo "  build            - Build Docker image"
	@echo "  run              - Run Docker container"
	@echo "  docker-compose-up - Start all services"
	@echo "  docker-compose-down - Stop all services"

help-dev:
	@echo "Development targets:"
	@echo "  setup-dev        - Setup development environment"
	@echo "  dev              - Run in development mode"
	@echo "  quick-start      - Quick start with all services"
	@echo "  stop             - Stop quick start services"
	@echo "  logs             - Show application logs"

# Default target
.DEFAULT_GOAL := help
