#!/bin/bash

# HackRx RAG Pipeline Startup Script (Redis Removed)
# This script starts the HackRx application without Redis dependencies

echo "🚀 Starting HackRx RAG Pipeline (Redis Removed)..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down

# Remove any existing containers and volumes (clean start)
echo "🧹 Cleaning up existing containers and volumes..."
docker-compose down -v --remove-orphans

# Build the application
echo "🔨 Building HackRx application..."
docker-compose build --no-cache

# Start the application
echo "🚀 Starting HackRx application..."
docker-compose up -d

# Wait for the application to be ready
echo "⏳ Waiting for application to be ready..."
sleep 30

# Check application health
echo "🏥 Checking application health..."
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Application is healthy and running!"
    echo "🌐 Access the application at: http://localhost:5000"
    echo "📊 Health check: http://localhost:5000/health"
    echo "🔍 API endpoint: http://localhost:5000/hackrx/run"
else
    echo "❌ Application health check failed. Checking logs..."
    docker-compose logs hackrx-app
    exit 1
fi

echo ""
echo "🎉 HackRx RAG Pipeline started successfully!"
echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose logs -f hackrx-app"
echo "  Stop app:  docker-compose down"
echo "  Restart:   docker-compose restart"
echo "  Status:    docker-compose ps"
echo ""
echo "🔗 Application URLs:"
echo "  Main app:  http://localhost:5000"
echo "  Health:    http://localhost:5000/health"
echo "  API:       http://localhost:5000/hackrx/run"
echo ""
echo "📚 For more information, see README.md and DOCKER_SETUP.md"