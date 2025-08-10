# Docker Setup for HackRx Project

## Overview
This Docker setup provides a complete containerized environment for the HackRx document processing pipeline with Redis caching.

## Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for containers
- Internet connection for downloading models

## Quick Start

### 1. Build and Run with Docker Compose
```bash
# Clone and navigate to project directory
cd hackrx_final_x2X

# Copy environment file and customize if needed
cp env.example .env

# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### 2. Access the Application
- **API Endpoint**: http://localhost:5000/hackrx/run
- **Health Check**: http://localhost:5000/health
- **Redis**: localhost:6379

### 3. Test the API
```bash
curl -X POST http://localhost:5000/hackrx/run \
  -H "Authorization: Bearer 9f40f077e610d431226b59eec99652153ccad94769da6779cc01725731999634" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

## Services

### HackRx Application (hackrx-app)
- **Port**: 5000
- **Health Check**: `/health` endpoint
- **Features**:
  - Document processing pipeline
  - Redis caching integration
  - LLM integrations (NVIDIA, Groq)
  - Vector search with FAISS
  - Session management

### Redis Cache (redis)
- **Port**: 6379
- **Memory Limit**: 512MB
- **Persistence**: Enabled with AOF
- **Eviction Policy**: allkeys-lru

## Environment Variables

### Required API Keys
```bash
NVIDIA_API_KEY=your_nvidia_api_key
GROQ_API_KEY=your_groq_api_key
API_BEARER_TOKEN=your_bearer_token
```

### Redis Configuration
```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
```

### Application Configuration
```bash
EMBEDDING_MODEL=intfloat/e5-base-v2
FLASK_ENV=production
PORT=5000
```

## Volume Mounts

The following directories are mounted for data persistence:
- `./processing_results` - Processing outputs
- `./document_processing/logs` - Application logs
- `./document_processing/metadata` - Document metadata
- `./document_processing/embeddings` - Cached embeddings
- `./vectorstore` - FAISS indices

## Docker Commands

### Build Only
```bash
docker-compose build
```

### Start Services
```bash
docker-compose up -d
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f hackrx-app
docker-compose logs -f redis
```

### Stop Services
```bash
docker-compose down
```

### Stop and Remove Volumes
```bash
docker-compose down -v
```

### Restart Services
```bash
docker-compose restart
```

## Health Checks

Both services include health checks:
- **Application**: Checks `/health` endpoint every 30s
- **Redis**: Checks `redis-cli ping` every 30s

View health status:
```bash
docker-compose ps
```

## Troubleshooting

### Memory Issues
If you encounter memory issues:
1. Increase Docker memory limit (4GB+ recommended)
2. Reduce Redis memory limit in docker-compose.yml
3. Consider using smaller embedding models

### Port Conflicts
If port 5000 or 6379 are in use:
```bash
# Change ports in docker-compose.yml
ports:
  - "5001:5000"  # Application
  - "6380:6379"  # Redis
```

### Model Download Issues
First run may take time downloading models:
```bash
# Monitor download progress
docker-compose logs -f hackrx-app
```

### Redis Connection Issues
Check Redis connectivity:
```bash
docker-compose exec redis redis-cli ping
```

### Application Logs
Check detailed application logs:
```bash
docker-compose exec hackrx-app tail -f /app/document_processing/logs/app.log
```

## Performance Optimization

### For Production
1. Use multi-stage Docker build
2. Implement proper logging aggregation
3. Set up monitoring with Prometheus/Grafana
4. Use external Redis cluster for scaling
5. Configure nginx reverse proxy

### For Development
```bash
# Override with development settings
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Security Considerations

1. **API Keys**: Store in environment variables or secrets
2. **Bearer Token**: Use strong, unique tokens
3. **Redis**: Enable password authentication in production
4. **Network**: Use custom networks in production
5. **Volumes**: Set proper file permissions

## Scaling

### Horizontal Scaling
```bash
# Scale application instances
docker-compose up --scale hackrx-app=3
```

### Load Balancer (nginx)
Add nginx service to docker-compose.yml for load balancing multiple app instances.

## Backup and Recovery

### Backup Redis Data
```bash
docker-compose exec redis redis-cli BGSAVE
docker cp hackrx-redis:/data/dump.rdb ./backup/
```

### Backup Application Data
```bash
tar -czf backup-$(date +%Y%m%d).tar.gz \
  processing_results/ \
  document_processing/logs/ \
  document_processing/metadata/ \
  document_processing/embeddings/ \
  vectorstore/
```

## Monitoring

### Container Stats
```bash
docker stats hackrx-app hackrx-redis
```

### Application Metrics
The application exposes health metrics at `/health` endpoint.

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify health: `docker-compose ps`
3. Test connectivity: `curl http://localhost:5000/health`
4. Check Redis: `docker-compose exec redis redis-cli ping`