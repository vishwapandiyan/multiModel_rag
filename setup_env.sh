#!/bin/bash

# Setup Environment File Script
# This script helps you create a .env file from the template

echo "🔧 HackRx Environment Setup"
echo "=========================="

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Setup cancelled."
        exit 1
    fi
fi

# Copy template to .env
if [ -f "env.template" ]; then
    cp env.template .env
    echo "✅ Created .env from template"
else
    echo "❌ env.template not found!"
    exit 1
fi

echo ""
echo "📝 Environment file created at: .env"
echo ""
echo "🔐 IMPORTANT SECURITY NOTES:"
echo "=============================="
echo "1. The .env file will be committed to git (as per your request)"
echo "2. Current API keys are demo keys - replace with your own"
echo "3. For production, use environment variables instead"
echo "4. Never commit real secrets to public repositories"
echo ""
echo "🚀 Next steps:"
echo "1. Edit .env with your actual API keys (if needed)"
echo "2. Run: docker-compose up --build -d"
echo "3. Test: curl http://localhost:5000/health"
echo ""
echo "✅ Setup complete!"