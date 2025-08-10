#!/bin/bash

# HackRx Docker Quick Start Script

set -e  # Exit on any error

echo "🐳 HackRx Docker Quick Start"
echo "=============================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker is running"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created. You can customize API keys if needed."
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p processing_results
mkdir -p document_processing/logs
mkdir -p document_processing/metadata
mkdir -p document_processing/embeddings
mkdir -p vectorstore/faiss_index
mkdir -p retriever/vectorstore/faiss_index

echo "✅ Directories created"

# Build and start services
echo "🏗️ Building and starting Docker containers..."
echo "   This may take a few minutes on first run (downloading models)..."

docker-compose up --build -d

echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are starting up"
    
    echo "🧪 Running integration tests..."
    python3 test_docker.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Your HackRx application is running!"
        echo ""
        echo "📍 Access Points:"
        echo "   • API Endpoint: http://localhost:5000/hackrx/run"
        echo "   • Health Check: http://localhost:5000/health"
        echo "   • Redis: localhost:6379"
        echo ""
        echo "🔐 Authentication:"
        echo "   Bearer Token: 9f40f077e610d431226b59eec99652153ccad94769da6779cc01725731999634"
        echo ""
        echo "📋 Useful Commands:"
        echo "   • View logs: docker-compose logs -f"
        echo "   • Stop services: docker-compose down"
        echo "   • Restart: docker-compose restart"
        echo ""
        echo "📖 For detailed documentation, see DOCKER_SETUP.md"
    else
        echo "⚠️ Services started but tests failed. Check logs:"
        echo "   docker-compose logs"
    fi
else
    echo "❌ Services failed to start. Check logs:"
    echo "   docker-compose logs"
    exit 1
fi