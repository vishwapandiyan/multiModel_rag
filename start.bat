@echo off
REM HackRx RAG Pipeline Startup Script for Windows (Redis Removed)
REM This script starts the HackRx application without Redis dependencies

echo 🚀 Starting HackRx RAG Pipeline (Redis Removed)...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ docker-compose is not installed. Please install it and try again.
    pause
    exit /b 1
)

REM Stop any existing containers
echo 🛑 Stopping any existing containers...
docker-compose down

REM Remove any existing containers and volumes (clean start)
echo 🧹 Cleaning up existing containers and volumes...
docker-compose down -v --remove-orphans

REM Build the application
echo 🔨 Building HackRx application...
docker-compose build --no-cache

REM Start the application
echo 🚀 Starting HackRx application...
docker-compose up -d

REM Wait for the application to be ready
echo ⏳ Waiting for application to be ready...
timeout /t 30 /nobreak >nul

REM Check application health
echo 🏥 Checking application health...
curl -f http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Application health check failed. Checking logs...
    docker-compose logs hackrx-app
    pause
    exit /b 1
) else (
    echo ✅ Application is healthy and running!
    echo 🌐 Access the application at: http://localhost:5000
    echo 📊 Health check: http://localhost:5000/health
    echo 🔍 API endpoint: http://localhost:5000/hackrx/run
)

echo.
echo 🎉 HackRx RAG Pipeline started successfully!
echo.
echo 📋 Useful commands:
echo   View logs: docker-compose logs -f hackrx-app
echo   Stop app:  docker-compose down
echo   Restart:   docker-compose restart
echo   Status:    docker-compose ps
echo.
echo 🔗 Application URLs:
echo   Main app:  http://localhost:5000
echo   Health:    http://localhost:5000/health
echo   API:       http://localhost:5000/hackrx/run
echo.
echo 📚 For more information, see README.md and DOCKER_SETUP.md
pause
