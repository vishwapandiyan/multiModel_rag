"""
HackRx Final Pipeline - Main Integration Module

Production-ready document processing and AI challenge-solving system.
"""

import os
import sys
import json
import time
import logging
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import requests
from flask import Flask, request, jsonify
import faiss
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_extraction.extractor import TextExtractor
from document_processing.embedding.embed_generator import EmbeddingGenerator
from document_processing.chunks.chunk_splitter import ChunkSplitter
from document_processing.metadata.metadata_extractor import MetadataExtractor
from document_processing.logs.logging_config import setup_logging, get_logger
from document_processing.prompts import get_prompt
from vectorstore.vector_store import VectorStore
from retriever.retriever import Retriever
from agent.new_agent_simple import SimpleWorkflowAgent, LangGraphGenericAgent, LANGGRAPH_AVAILABLE

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from redis_integration.llm_cache import LLMCache
    from redis_integration.session_manager import SessionManager
    from redis_integration.metadata_cache import MetadataCache
    from redis_integration.preprocessing_cache import PreprocessingCache
    REDIS_CACHE_AVAILABLE = True
except ImportError:
    REDIS_CACHE_AVAILABLE = False
setup_logging()
logger = get_logger(__name__)


NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "9f40f077e610d431226b59eec99652153ccad94769da6779cc01725731999634")

app = Flask(__name__)
text_extractor = None
embedding_generator = None
chunk_splitter = None
metadata_extractor = None
vector_store = None
retriever = None
challenge_agent = None


llm_cache = None
session_manager = None
metadata_cache = None
preprocessing_cache = None

def initialize_components():
    """Initialize all pipeline components"""
    global text_extractor, embedding_generator, chunk_splitter, metadata_extractor
    global vector_store, retriever, challenge_agent
    global llm_cache, session_manager, metadata_cache, preprocessing_cache
    
    try:
        logger.info("Initializing pipeline components...")
        
        # Initialize text extraction
        text_extractor = TextExtractor()
        logger.info("Text extractor initialized")
        
        # Initialize embedding generation with same model as api_demo_new.py
        embedding_generator = EmbeddingGenerator(model_name=EMBEDDING_MODEL)
        logger.info(f"Embedding generator initialized with model: {EMBEDDING_MODEL}")
        
        # Initialize chunking
        chunk_splitter = ChunkSplitter(chunk_size=1000, chunk_overlap=300)
        logger.info("Chunk splitter initialized")
        
        # Initialize metadata extraction
        metadata_extractor = MetadataExtractor()
        logger.info("Metadata extractor initialized")
        
        # Initialize vector store
        vector_store = VectorStore()
        logger.info("Vector store initialized")
        
        # Initialize retriever
        retriever = Retriever(embedding_model=EMBEDDING_MODEL)
        logger.info("Retriever initialized")
        
        # Initialize challenge agent
        try:
            if LANGGRAPH_AVAILABLE:
                challenge_agent = LangGraphGenericAgent()
                logger.info("LangGraph challenge agent initialized")
            else:
                challenge_agent = SimpleWorkflowAgent()
                logger.info("Simple workflow challenge agent initialized")
        except Exception as agent_error:
            logger.warning(f"Error initializing LangGraph agent: {agent_error}")
            # Fallback to simple workflow agent
            challenge_agent = SimpleWorkflowAgent()
            logger.info("Fallback to simple workflow challenge agent")
        
        # Initialize Redis cache components
        if REDIS_CACHE_AVAILABLE:
            try:
                llm_cache = LLMCache()
                session_manager = SessionManager()
                metadata_cache = MetadataCache()
                preprocessing_cache = PreprocessingCache()
                logger.info("✅ Redis cache components initialized")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False

def get_nvidia_llm():
    """Get NVIDIA ChatNVIDIA instance"""
    if not NVIDIA_AVAILABLE:
        return None
    
    try:
        return ChatNVIDIA(
            model="meta/llama-3.1-70b-instruct",
            api_key=NVIDIA_API_KEY,
            temperature=0.7,
            max_tokens=1024
        )
    except Exception as e:
        logger.error(f"Error initializing NVIDIA LLM: {e}")
        return None

def get_groq_client():
    """Get Groq client instance"""
    if not GROQ_AVAILABLE:
        return None
    
    try:
        return Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.error(f"Error initializing Groq client: {e}")
        return None

def detect_puzzle_document(document_text: str, sampled_pages: Optional[str] = None, questions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect if document is a puzzle/challenge document using both rule-based and LLM analysis
    
    Args:
        document_text: Full document text
        sampled_pages: Optional sampled pages
        questions: Optional questions for context
        
    Returns:
        Dictionary with puzzle detection results
    """
    try:
        if sampled_pages is None:
            sampled_pages = chunk_splitter.sample_random_pages(document_text)
        
        # Rule-based detection
        puzzle_keywords = [
            'challenge', 'puzzle', 'mission', 'task', 'solve', 'find',
            'flight', 'city', 'endpoint', 'api', 'hackrx', 'register',
            'teams', 'public', 'flights', 'gateway', 'taj mahal',
            'eiffel tower', 'big ben', 'marina beach'
        ]
        
        text_lower = sampled_pages.lower()
        keyword_matches = [keyword for keyword in puzzle_keywords if keyword in text_lower]
        
        # Check for specific patterns
        has_flight_patterns = any(pattern in text_lower for pattern in ['flight', 'city', 'endpoint'])
        has_api_patterns = any(pattern in text_lower for pattern in ['api', 'endpoint', 'register.hackrx.in'])
        
        rule_based_confidence = len(keyword_matches) / len(puzzle_keywords)
        rule_based_is_puzzle = len(keyword_matches) > 3 or (has_flight_patterns and has_api_patterns)
        
        # LLM-based detection for additional validation
        llm_result = {'is_puzzle': rule_based_is_puzzle, 'confidence': rule_based_confidence}
        
        try:
            nvidia_llm = get_nvidia_llm()
            if nvidia_llm:
                prompt = get_prompt('puzzle_detection', document_sample=sampled_pages[:2000])
                response = nvidia_llm.invoke(prompt)
                
                import json
                import re
                
                # Extract JSON from response
                response_content = response.content if hasattr(response, 'content') else str(response)
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                
                if json_match:
                    llm_result = json.loads(json_match.group())
                    
        except Exception as llm_error:
            logger.warning(f"LLM puzzle detection failed, using rule-based result: {llm_error}")
        
        # Combine rule-based and LLM results
        final_confidence = max(rule_based_confidence, llm_result.get('confidence', 0))
        final_is_puzzle = rule_based_is_puzzle or llm_result.get('is_puzzle', False)
        
        return {
            'is_puzzle': final_is_puzzle,
            'confidence': final_confidence,
            'keywords_found': keyword_matches,
            'has_flight_patterns': has_flight_patterns,
            'has_api_patterns': has_api_patterns,
            'llm_analysis': llm_result,
            'rule_based_confidence': rule_based_confidence
        }
        
    except Exception as e:
        logger.error(f"Error detecting puzzle document: {e}")
        return {
            'is_puzzle': False,
            'confidence': 0.0,
            'keywords_found': [],
            'has_flight_patterns': False,
            'has_api_patterns': False
        }

def process_document_with_pipeline(url: str, questions: List[str]) -> Dict[str, Any]:
    """
    Process document through the complete pipeline
    
    Args:
        url: Document URL
        questions: List of questions to answer
        
    Returns:
        Processing results
    """
    try:
        start_time = time.time()
        logger.info(f"Starting document processing for URL: {url}")
        
        # Generate URL hash for tracking
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        
        document_text = None
        if preprocessing_cache:
            document_text = preprocessing_cache.get_extracted_text(url)
            if document_text:
                logger.info("Retrieved extracted text from cache")
        
        if not document_text:
            # Download and extract text
            logger.info("Downloading and extracting text...")
            download_path = f"temp_{url_hash}.tmp"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                with open(download_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract text based on file type
                document_text = text_extractor.extract_from_file(download_path)
                
                if not document_text:
                    return {
                        "success": False,
                        "error": "Failed to extract text from document",
                        "processing_time": time.time() - start_time
                    }
                
                if preprocessing_cache:
                    preprocessing_cache.set_extracted_text(url, document_text)
                
            finally:
                # Cleanup temp file
                if os.path.exists(download_path):
                    os.remove(download_path)
        
        logger.info(f"Extracted {len(document_text)} characters from document")
        
        # Extract metadata
        logger.info("Extracting metadata...")
        metadata = metadata_extractor.extract_basic_metadata(url, document_text)
        metadata_extractor.save_metadata(url_hash, metadata)
        
        logger.info("Checking for puzzle/challenge content...")
        puzzle_detection = None
        if preprocessing_cache:
            puzzle_detection = preprocessing_cache.get_puzzle_detection_result(document_text, questions)
            if puzzle_detection:
                logger.info("Retrieved puzzle detection result from cache")
        
        if not puzzle_detection:
            puzzle_detection = detect_puzzle_document(document_text, questions=questions)
            if preprocessing_cache:
                preprocessing_cache.set_puzzle_detection_result(document_text, puzzle_detection, questions)
        
        # Route to appropriate processing pipeline
        if puzzle_detection['is_puzzle'] and puzzle_detection['confidence'] > 0.5:
            logger.info("Detected puzzle document, routing to challenge agent...")
            
            # Use challenge agent for puzzle documents
            result = challenge_agent.process_document_with_questions(
                document_text, 
                questions,
                puzzle_type="challenge",
                confidence=puzzle_detection['confidence']
            )
            
            # Add metadata to result
            result['puzzle_detection'] = puzzle_detection
            result['processing_method'] = 'challenge_agent'
            result['processing_time'] = time.time() - start_time
            
            return result
        
        else:
            logger.info("Processing as standard document...")
            
            # Standard document processing pipeline
            
            # Chunk the text with caching
            logger.info("Chunking text...")
            chunks = None
            if preprocessing_cache:
                chunks = preprocessing_cache.get_text_chunks(
                    document_text, 
                    chunk_splitter.chunk_size, 
                    chunk_splitter.chunk_overlap
                )
                if chunks:
                    logger.info(f"Retrieved {len(chunks)} chunks from cache")
            
            if not chunks:
                chunks = chunk_splitter.chunk_text(document_text)
                # Cache the chunks
                if preprocessing_cache:
                    preprocessing_cache.set_text_chunks(
                        document_text, chunks, 
                        chunk_splitter.chunk_size, 
                        chunk_splitter.chunk_overlap
                    )
                logger.info(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = embedding_generator.generate_embeddings(chunks)
            
            if len(embeddings) == 0:
                return {
                    "success": False,
                    "error": "Failed to generate embeddings",
                    "processing_time": time.time() - start_time
                }
            
            # Create/update vector store
            logger.info("Updating vector store...")
            vector_store.add_documents(chunks, embeddings, url_hash)
            
            # Process questions
            logger.info("Processing questions...")
            answers = []
            
            for question in questions:
                try:
                    # Retrieve relevant chunks using FAISS semantic search
                    search_results = retriever.faiss_semantic_search(
                        vector_store.index, chunks, question, top_k=5
                    )
                    
                    # Convert search results to chunks with content
                    relevant_chunks = []
                    for idx, score in search_results:
                        if idx < len(chunks):
                            relevant_chunks.append({
                                'content': chunks[idx],
                                'score': score,
                                'index': idx
                            })
                    
                    # Generate answer using NVIDIA LLM with caching
                    nvidia_llm = get_nvidia_llm()
                    if nvidia_llm and relevant_chunks:
                        context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
                        
                        prompt = get_prompt('enhanced_answer', question=question, context=context)
                        
                        # Try to get from cache first (only for document-specific Q&A)
                        answer = None
                        if llm_cache:
                            # Only cache factual document-based answers, not creative responses
                            answer = llm_cache.get_enhanced_answer_response(question, context)
                            if answer:
                                logger.info("Retrieved NVIDIA LLM response from cache")
                        
                        if not answer:
                            try:
                                response = nvidia_llm.invoke(prompt)
                                answer = response.content if hasattr(response, 'content') else str(response)
                                
                                # Cache the response (only factual document-based answers)
                                if llm_cache and answer:
                                    llm_cache.set_enhanced_answer_response(question, context, answer)
                                    
                            except Exception as llm_error:
                                logger.warning(f"NVIDIA LLM failed, using fallback: {llm_error}")
                                answer = f"Based on the document content: {context[:500]}..."
                    else:
                        answer = "Unable to generate answer - no relevant content found or LLM unavailable"
                    
                    answers.append(answer)
                    
                except Exception as q_error:
                    logger.error(f"Error processing question '{question}': {q_error}")
                    answers.append(f"Error processing question: {q_error}")
            
            # Save processing results
            logger.info("Saving processing results...")
            processing_data = {
                'url': url,
                'url_hash': url_hash,
                'questions': questions,
                'answers': answers,
                'chunks': chunks,
                'metadata': metadata,
                'puzzle_detection': puzzle_detection,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to disk
            results_dir = Path("processing_results")
            results_dir.mkdir(exist_ok=True)
            
            with open(results_dir / f"{url_hash}_results.json", 'w', encoding='utf-8') as f:
                json.dump(processing_data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "answers": answers,
                "metadata": metadata,
                "puzzle_detection": puzzle_detection,
                "processing_method": "standard_pipeline",
                "processing_time": time.time() - start_time,
                "chunks_processed": len(chunks),
                "embeddings_generated": len(embeddings)
            }
    
    except Exception as e:
        logger.error(f"Error in document processing pipeline: {e}")
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "text_extractor": text_extractor is not None,
            "embedding_generator": embedding_generator is not None,
            "chunk_splitter": chunk_splitter is not None,
            "metadata_extractor": metadata_extractor is not None,
            "vector_store": vector_store is not None,
            "retriever": retriever is not None,
            "challenge_agent": challenge_agent is not None,
            "nvidia_llm": NVIDIA_AVAILABLE,
            "groq_client": GROQ_AVAILABLE,
            "langgraph": LANGGRAPH_AVAILABLE
        }
    })

@app.route('/hackrx/run', methods=['POST'])
def process_document():
    """Main document processing endpoint with session management and caching"""
    try:
        # Validate authentication with specific Bearer token
        auth_header = request.headers.get('Authorization', '')
        expected_token = f'Bearer {API_BEARER_TOKEN}'
        
        if auth_header != expected_token:
            return jsonify({
                "success": False,
                "error": "Missing or invalid authorization token"
            }), 401
        
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        # Extract parameters
        document_url = data.get('documents')
        questions = data.get('questions', [])
        
        if not document_url:
            return jsonify({
                "success": False,
                "error": "Missing 'documents' parameter"
            }), 400
        
        if not questions:
            return jsonify({
                "success": False,
                "error": "Missing 'questions' parameter"
            }), 400
        
        # Session management
        session_id = request.headers.get('X-Session-ID')
        if not session_id and session_manager:
            session_id = session_manager.create_session()
        elif session_id and session_manager:
            session_manager.increment_request_count(session_id)
        
        # Check request cache
        cached_result = None
        if session_manager:
            request_key = session_manager.generate_request_key(document_url, questions)
            cached_result = session_manager.get_cached_request_result(request_key)
            
            if cached_result:
                logger.info("Retrieved result from request cache")
                # Add session info to response
                cached_result['session_id'] = session_id
                return jsonify(cached_result)
        
        # Process document
        logger.info(f"Processing request for document: {document_url}")
        result = process_document_with_pipeline(document_url, questions)
        
        # Cache the result
        if session_manager and result.get('success'):
            session_manager.cache_request_result(request_key, result)
            
            # Add processed document to session
            if session_id:
                url_hash = hashlib.sha256(document_url.encode()).hexdigest()[:16]
                session_manager.add_processed_document(session_id, url_hash, document_url)
        
        # Add session info to response
        result['session_id'] = session_id
        
        # Return result
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /hackrx/run endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List processed documents"""
    try:
        # Get all processed documents
        results_dir = Path("processing_results")
        if not results_dir.exists():
            return jsonify({
                "success": True,
                "documents": [],
                "count": 0
            })
        
        documents = []
        for result_file in results_dir.glob("*_results.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                documents.append({
                    "url_hash": data.get('url_hash'),
                    "url": data.get('url'),
                    "timestamp": data.get('timestamp'),
                    "processing_time": data.get('processing_time'),
                    "questions_count": len(data.get('questions', [])),
                    "chunks_processed": data.get('chunks_processed', 0),
                    "is_puzzle": data.get('puzzle_detection', {}).get('is_puzzle', False)
                })
            except Exception as file_error:
                logger.warning(f"Error reading result file {result_file}: {file_error}")
        
        # Sort by timestamp (newest first)
        documents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            "success": True,
            "documents": documents,
            "count": len(documents)
        })
        
    except Exception as e:
        logger.error(f"Error in /documents endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/documents/<url_hash>', methods=['GET'])
def get_document_details(url_hash):
    """Get details for a specific document"""
    try:
        results_dir = Path("processing_results")
        result_file = results_dir / f"{url_hash}_results.json"
        
        if not result_file.exists():
            return jsonify({
                "success": False,
                "error": "Document not found"
            }), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify({
            "success": True,
            "document": data
        })
        
    except Exception as e:
        logger.error(f"Error in /documents/{url_hash} endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

def main():
    """Main function to run the application"""
    try:
        logger.info("Starting HackRx Final Pipeline...")
        
        # Initialize components
        if not initialize_components():
            logger.error("Failed to initialize components")
            return False
        
        # Create necessary directories
        os.makedirs("processing_results", exist_ok=True)
        os.makedirs("document_processing/logs", exist_ok=True)
        os.makedirs("document_processing/metadata", exist_ok=True)
        os.makedirs("vectorstore/faiss_index", exist_ok=True)
        
        logger.info("All directories created/verified")
        
        # Test the system with a simple check
        logger.info("Running system health check...")
        
        # Test text extraction
        test_text = "This is a test document for validation."
        test_chunks = chunk_splitter.chunk_text(test_text)
        test_embeddings = embedding_generator.generate_embeddings(test_chunks)
        
        if len(test_embeddings) > 0:
            logger.info("✅ System health check passed")
        else:
            logger.warning("⚠️ System health check failed - no embeddings generated")
        
        logger.info("🚀 HackRx Final Pipeline is ready!")
        logger.info("Available endpoints:")
        logger.info("  - POST /hackrx/run - Process documents")
        logger.info("  - GET /health - Health check")
        logger.info("  - GET /documents - List processed documents")
        logger.info("  - GET /documents/<hash> - Get document details")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in main initialization: {e}")
        return False

if __name__ == "__main__":
    # Initialize the system
    if main():
        # Run the Flask app
        # Get port from environment variable (useful for cloud deployments)
        port = int(os.getenv("PORT", 5000))
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            threaded=True
        )
    else:
        logger.error("Failed to start the application")
        sys.exit(1)
