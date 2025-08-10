# HackRx Final Pipeline

A comprehensive, modular document processing and challenge-solving system that combines advanced text extraction, embedding generation, vector search, and agent-based puzzle solving.

## 🏗️ Project Structure

```
hackrx_final_x2X/
│
├── text_extraction/                 # Text extraction from multiple formats
│   ├── __init__.py
│   └── extractor.py                 # Main text extraction class
│
├── document_processing/              # Document preprocessing pipeline
│   ├── embedding/                   # Embedding generation
│   │   ├── __init__.py
│   │   └── embed_generator.py       # SentenceTransformer embeddings
│   │
│   ├── chunks/                      # Text chunking and splitting
│   │   ├── __init__.py
│   │   └── chunk_splitter.py        # Intelligent text chunking
│   │
│   ├── logs/                        # Logging configuration
│   │   ├── app.log                  # Application logs
│   │   └── logging_config.py        # Centralized logging setup
│   │
│   └── metadata/                    # Metadata extraction
│       ├── __init__.py
│       └── metadata_extractor.py    # Document metadata handling
│
├── retriever/                       # Document retrieval
│   ├── __init__.py
│   └── retriever.py                 # Semantic search and retrieval
│
├── vectorstore/                     # Vector database (FAISS)
│   ├── __init__.py
│   ├── vector_store.py              # FAISS vector operations
│   └── faiss_index/                 # FAISS index storage
│       └── index.faiss
│
├── agent/                           # Challenge solving agents
│   ├── __init__.py
│   └── new_agent_simple.py          # LangGraph-based challenge agents
│
├── main.py                          # Main integration and API
├── requirements.txt                 # Python dependencies
└── README.md                        # This documentation
```

## 🚀 Features

### 📄 Text Extraction
- **Multi-format support**: PDF, Word, Excel, PowerPoint, Images, HTML, ZIP, Binary
- **OCR capabilities**: Automatic text extraction from images and scanned documents
- **Intelligent parsing**: Format-specific extraction with structure preservation
- **Malicious content detection**: Basic security checks for extracted content

### 🧠 Document Processing
- **Smart chunking**: Intelligent text splitting with semantic boundary detection
- **Advanced embeddings**: SentenceTransformer-based embeddings (`intfloat/e5-base-v2`)
- **Metadata extraction**: Comprehensive document metadata and analysis
- **Centralized logging**: Structured logging with performance monitoring

### 🔍 Vector Search & Retrieval
- **FAISS integration**: High-performance similarity search with GPU support
- **Multiple index types**: Flat, IVF, and HNSW indexes
- **Semantic retrieval**: Context-aware document chunk retrieval
- **Relevance filtering**: Smart filtering of irrelevant content

### 🤖 Agent-Based Challenge Solving
- **LangGraph workflows**: Complex multi-step challenge solving
- **Puzzle detection**: Automatic identification of challenge documents
- **API integration**: Dynamic endpoint discovery and interaction
- **Fallback workflows**: Simple workflow agents when LangGraph unavailable

### 🌐 REST API
- **Document processing**: `/hackrx/run` endpoint for complete pipeline
- **Answers only**: `/hackrx/answers` endpoint for answers without metadata
- **Health monitoring**: `/health` endpoint with component status
- **Document management**: `/documents` endpoints for tracking processed files
- **Authentication**: Bearer token-based security

## 📡 API Usage

### Response Formats

The system supports two response formats:

1. **Full Response** (`/hackrx/run` or `format: "full"`):
   ```json
   {
     "success": true,
     "answers": ["Answer 1", "Answer 2"],
     "metadata": {...},
     "puzzle_detection": {...},
     "processing_method": "standard_pipeline",
     "processing_time": 1.23,
     "chunks_processed": 10,
     "embeddings_generated": 10
   }
   ```

2. **Answers Only** (`/hackrx/answers` or `format: "answers_only"`):
   ```json
   {
     "success": true,
     "answers": ["Answer 1", "Answer 2"]
   }
   ```

### Example Requests

**Get full response:**
```bash
curl -X POST "http://localhost:5000/hackrx/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic?"]
  }'
```

**Get answers only:**
```bash
curl -X POST "http://localhost:5000/hackrx/answers" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic?"]
  }'
```

**Get answers only from main endpoint:**
```bash
curl -X POST "http://localhost:5000/hackrx/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic?"],
    "format": "answers_only"
  }'
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Tesseract OCR (for image text extraction)

### Quick Setup

1. **Clone and navigate to the project**:
   ```bash
   cd hackrx_final_x2X
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**:
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Set up API keys** (edit `main.py`):
   ```python
   GROQ_API_KEY = "your-groq-api-key"
   ```

## 🚀 Usage

### Starting the Application

```bash
python main.py
```

The application will start on `http://localhost:5000` with the following endpoints:

### API Endpoints

#### 1. Process Document
```bash
POST /hackrx/run
```

**Headers**:
```
Authorization: Bearer your-token
Content-Type: application/json
```

**Request Body**:
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the main topic of this document?",
    "What are the key findings?"
  ]
}
```

**Response**:
```json
{
  "success": true,
  "answers": [
    "The main topic is...",
    "The key findings are..."
  ],
  "processing_method": "standard_pipeline",
  "processing_time": 15.32,
  "puzzle_detection": {
    "is_puzzle": false,
    "confidence": 0.1
  }
}
```

#### 2. Health Check
```bash
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "components": {
    "text_extractor": true,
    "embedding_generator": true,
    "vector_store": true,
    "challenge_agent": true
  }
}
```

#### 3. List Documents
```bash
GET /documents
```

#### 4. Get Document Details
```bash
GET /documents/{url_hash}
```

### Programming Interface

```python
from main import initialize_components, process_document_with_pipeline

# Initialize the system
initialize_components()

# Process a document
result = process_document_with_pipeline(
    url="https://example.com/document.pdf",
    questions=["What is this document about?"]
)

print(result)
```

## 🧩 Challenge Solving

The system automatically detects puzzle/challenge documents and routes them to specialized agents:

### Supported Challenge Types
- **Flight puzzles**: HackRx-style flight number challenges
- **API challenges**: Dynamic endpoint discovery and interaction
- **Data extraction**: Complex information retrieval tasks

### Challenge Detection
The system looks for keywords and patterns:
- Challenge-related terms: "puzzle", "mission", "challenge"
- API patterns: "endpoint", "register.hackrx.in"
- Flight patterns: "flight number", "city", "landmark"

### Agent Workflows
1. **Document Analysis**: Extract challenge structure and rules
2. **Entity Retrieval**: Get initial data from APIs
3. **Entity Mapping**: Map entities to targets using rules
4. **Endpoint Selection**: Choose correct API endpoints
5. **Result Fetching**: Retrieve final answers
6. **Answer Generation**: Compile structured responses

## 🔧 Configuration

### Embedding Models
The system uses `intfloat/e5-base-v2` by default. To change:

```python
# In main.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Alternative model
```

### Chunking Parameters
```python
# In main.py or when initializing ChunkSplitter
chunk_splitter = ChunkSplitter(
    chunk_size=1000,    # Maximum chunk size
    chunk_overlap=300   # Overlap between chunks
)
```

### Vector Store Settings
```python
# In vector_store.py
vector_store = VectorStore(
    index_type="flat",  # "flat", "ivf", "hnsw"
    metric="ip"         # "ip" (inner product), "l2"
)
```

## 📊 Performance

### Benchmarks
- **Text extraction**: ~2-5 seconds per document
- **Embedding generation**: ~0.1 seconds per chunk
- **Vector search**: ~0.01 seconds per query
- **Challenge solving**: ~5-15 seconds per puzzle

### Optimization Tips
1. **Use GPU**: Install `faiss-gpu` for faster similarity search
2. **Batch processing**: Process multiple documents together
3. **Index optimization**: Use IVF or HNSW for large document collections
4. **Caching**: Enable embedding caching for repeated documents

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Manual Testing
```bash
# Test individual components
python -c "from text_extraction.extractor import TextExtractor; print('Text extractor works!')"

# Test full pipeline
curl -X POST http://localhost:5000/health
```

### Example Documents
Test with various document types:
- PDF: Research papers, reports
- Word: Articles, documentation
- Excel: Data sheets, tables
- PowerPoint: Presentations
- Images: Screenshots, scanned documents

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install black flake8 pytest
   ```
4. Run tests and linting:
   ```bash
   black .
   flake8 .
   pytest
   ```

### Code Style
- Use Black for formatting
- Follow PEP 8 guidelines
- Add type hints where possible
- Document all public functions

## 📝 Logging

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARNING**: Important events that might need attention
- **ERROR**: Error conditions
- **CRITICAL**: Serious errors that might stop the application

### Log Files
- `document_processing/logs/app.log`: Main application log
- `document_processing/logs/performance.log`: Performance metrics

### Custom Logging
```python
from document_processing.logs.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Custom log message")
```

## 🚨 Troubleshooting

### Common Issues

1. **Tesseract not found**:
   ```bash
   # Install Tesseract OCR
   sudo apt-get install tesseract-ocr  # Ubuntu
   brew install tesseract              # macOS
   ```

2. **CUDA/GPU issues**:
   ```bash
   # Use CPU-only FAISS
   pip uninstall faiss-gpu
   pip install faiss-cpu
   ```

3. **Import errors**:
   ```bash
   # Add to PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:/path/to/hackrx_final_x2X"
   ```

4. **Memory issues with large documents**:
   - Reduce `chunk_size` in configuration
   - Process documents in smaller batches
   - Use IVF index for large vector collections

### Debug Mode
```python
# Enable debug logging
from document_processing.logs.logging_config import setup_logging
setup_logging(log_level="DEBUG")
```

## 📄 License

This project is part of the HackRx challenge and is intended for educational and demonstration purposes.

## 🙏 Acknowledgments

- **SentenceTransformers**: For high-quality embeddings
- **FAISS**: For efficient similarity search
- **LangGraph**: For agent workflow management
- **Flask**: For web API framework
- **PyMuPDF**: For PDF processing
- **Tesseract**: For OCR capabilities

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `document_processing/logs/`
3. Test individual components
4. Create an issue with detailed error information

---

*Built with ❤️ for the HackRx challenge*
