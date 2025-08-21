

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import faiss
try:
    from document_processing.embedding.embed_generator import EmbeddingGenerator
    from document_processing.prompts import get_prompt
    from retriever.prompts import get_retriever_prompt, RETRIEVER_PROMPT_CATEGORIES
except ImportError:
    from ..document_processing.embedding.embed_generator import EmbeddingGenerator
    from ..document_processing.prompts import get_prompt
    from .prompts import get_retriever_prompt, RETRIEVER_PROMPT_CATEGORIES

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Main document retrieval class with multiple search strategies"""
    
    def __init__(self, embedding_model: str = "intfloat/e5-base-v2"):
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.faiss_index = None
    
    def set_faiss_index(self, faiss_index):
        """Set FAISS index for fast similarity search"""
        self.faiss_index = faiss_index
    
    def faiss_semantic_search(self, index, chunks: List[str], question: str, 
                             top_k: int = 5) -> List[Tuple[int, float]]:
        """Perform semantic search using FAISS index"""
        try:
            if index is None:
                logger.warning("FAISS index not provided, falling back to enhanced search")
                return self.enhanced_semantic_search(None, chunks, question, top_k)
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search using FAISS
            distances, indices = index.search(query_embedding, top_k)
            
            # Convert to list of tuples (index, similarity_score)
            # FAISS returns distances, so we convert to similarity scores
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx != -1:  # FAISS returns -1 for invalid indices
                    # Convert distance to similarity score (1 - normalized_distance)
                    similarity = 1.0 - (distance / np.max(distances[0]))
                    results.append((int(idx), similarity))
            
            logger.info(f"FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS semantic search: {e}")
            # Fallback to enhanced search
            return self.enhanced_semantic_search(None, chunks, question, top_k)
    
    def enhanced_semantic_search(self, embeddings: Optional[np.ndarray], 
                                chunks: List[str], question: str, 
                                top_k: int = 5, faiss_index=None) -> List[Tuple[int, float]]:
        """Enhanced semantic search with multiple strategies"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            
            if faiss_index is not None:
                # Use FAISS if available
                return self.faiss_semantic_search(faiss_index, chunks, question, top_k)
            
            if embeddings is not None:
                # Use provided embeddings
                similarities = []
                for i, chunk_embedding in enumerate(embeddings):
                    similarity = self.embedding_generator.compute_similarity(
                        query_embedding, chunk_embedding
                    )
                    similarities.append((i, similarity))
                
                # Sort by similarity and return top-k
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
            else:
                # Generate embeddings on-the-fly
                chunk_embeddings = self.embedding_generator.generate_embeddings(chunks)
                return self.enhanced_semantic_search(chunk_embeddings, chunks, question, top_k)
                
        except Exception as e:
            logger.error(f"Error in enhanced semantic search: {e}")
            return []
    
    def check_relevance_enhanced(self, question: str, relevant_chunks: List[str]) -> Dict[str, Any]:
        """Enhanced relevance checking for retrieved chunks"""
        try:
            if not relevant_chunks:
                return {
                    'is_relevant': False,
                    'confidence': 0.0,
                    'reason': 'No chunks provided'
                }
            
            # Generate question embedding
            question_embedding = self.embedding_generator.generate_single_embedding(question)
            
            # Check relevance of each chunk
            chunk_scores = []
            for chunk in relevant_chunks:
                chunk_embedding = self.embedding_generator.generate_single_embedding(chunk)
                similarity = self.embedding_generator.compute_similarity(
                    question_embedding, chunk_embedding
                )
                chunk_scores.append(similarity)
            
            # Calculate overall relevance
            avg_similarity = np.mean(chunk_scores)
            max_similarity = np.max(chunk_scores)
            
            # Determine relevance threshold
            relevance_threshold = 0.25
            is_relevant = avg_similarity > relevance_threshold
            
            # Calculate confidence based on similarity distribution
            confidence = min(1.0, avg_similarity * 2)  # Scale to 0-1
            
            # Generate reason for relevance decision
            if is_relevant:
                if max_similarity > 0.8:
                    reason = "High similarity with question content"
                elif avg_similarity > 0.5:
                    reason = "Good overall similarity with question"
                else:
                    reason = "Moderate similarity, above threshold"
            else:
                reason = f"Low similarity ({avg_similarity:.3f}) below threshold ({relevance_threshold})"
            
            return {
                'is_relevant': is_relevant,
                'confidence': confidence,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'chunk_scores': chunk_scores,
                'reason': reason,
                'threshold': relevance_threshold
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced relevance check: {e}")
            return {
                'is_relevant': False,
                'confidence': 0.0,
                'reason': f'Error during relevance check: {str(e)}'
            }
    
    def process_question_with_relevance_check(self, question: str, embeddings: np.ndarray, 
                                            processed_chunks: List[str], 
                                            faiss_index=None) -> Dict[str, Any]:
        """Process question with relevance checking and retrieval"""
        try:
            # Perform semantic search
            if faiss_index is not None:
                search_results = self.faiss_semantic_search(
                    faiss_index, processed_chunks, question
                )
            else:
                search_results = self.enhanced_semantic_search(
                    embeddings, processed_chunks, question
                )
            
            # Extract relevant chunks
            relevant_chunks = []
            for idx, score in search_results:
                if idx < len(processed_chunks):
                    relevant_chunks.append(processed_chunks[idx])
            
            # Check relevance
            relevance_result = self.check_relevance_enhanced(question, relevant_chunks)
            
            return {
                'search_results': search_results,
                'relevant_chunks': relevant_chunks,
                'relevance_check': relevance_result,
                'question': question,
                'total_chunks_searched': len(processed_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing question with relevance check: {e}")
            return {
                'error': str(e),
                'question': question
            }
    
    def parallel_semantic_search(self, embeddings: np.ndarray, chunks: List[str], 
                                question: str, top_k: int = 5, 
                                max_workers: int = 6) -> List[Tuple[int, float]]:
        """Perform semantic search in parallel for large collections"""
        try:
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            
            # Split work into chunks for parallel processing
            chunk_size = max(1, len(chunks) // max_workers)
            chunk_indices = list(range(0, len(chunks), chunk_size))
            
            def process_chunk(start_idx):
                end_idx = min(start_idx + chunk_size, len(chunks))
                similarities = []
                for i in range(start_idx, end_idx):
                    similarity = self.embedding_generator.compute_similarity(
                        query_embedding, embeddings[i]
                    )
                    similarities.append((i, similarity))
                return similarities
            
            # Process chunks in parallel
            all_similarities = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {
                    executor.submit(process_chunk, start_idx): start_idx 
                    for start_idx in chunk_indices
                }
                
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_similarities = future.result()
                        all_similarities.extend(chunk_similarities)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {e}")
            
            # Sort by similarity and return top-k
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            return all_similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error in parallel semantic search: {e}")
            return []
    
    def expand_query_with_llm(self, original_question: str, num_variations: int = 4) -> List[str]:
        """Expand query using LLM for better retrieval"""
        try:
            # This would typically use an LLM to generate query variations
            # For now, we'll use simple heuristics
            
            variations = [original_question]
            
            # Add question variations based on common patterns
            if '?' in original_question:
                # Remove question mark for keyword search
                variations.append(original_question.replace('?', ''))
            
            # Add keyword variations
            words = original_question.lower().split()
            if len(words) > 3:
                # Use key terms
                key_terms = [word for word in words if len(word) > 3]
                if key_terms:
                    variations.append(' '.join(key_terms[:3]))
            
            # Add synonym variations (basic)
            synonym_map = {
                'find': ['locate', 'discover', 'identify'],
                'get': ['retrieve', 'obtain', 'fetch'],
                'show': ['display', 'present', 'reveal']
            }
            
            for word in words:
                if word in synonym_map:
                    for synonym in synonym_map[word]:
                        new_question = original_question.replace(word, synonym)
                        if new_question not in variations:
                            variations.append(new_question)
                            break
            
            # Limit to requested number of variations
            return variations[:num_variations]
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [original_question]
    
    def expand_query_with_enhanced_llm(self, original_question: str, num_variations: int = 4, llm=None) -> List[str]:
        """Enhanced query expansion using LLM with prompts from api_demo_new.py"""
        try:
            if llm is None:
                # Fallback to basic expansion
                return self.expand_query_with_llm(original_question, num_variations)
            
            # Use enhanced prompt for query expansion
            prompt = get_retriever_prompt(
                'query_expansion_enhanced',
                original_question=original_question,
                num_variations=num_variations
            )
            
            response = llm.invoke(prompt)
            
            # Try to extract JSON response
            import json
            try:
                start = response.content.find("[")
                end = response.content.rfind("]") + 1
                variations = json.loads(response.content[start:end])
                
                # Ensure original question is included
                if original_question not in variations:
                    variations.insert(0, original_question)
                
                return variations[:num_variations]
                
            except (json.JSONDecodeError, AttributeError):
                # Fallback to basic expansion
                logger.warning("Failed to parse LLM response for query expansion, using fallback")
                return self.expand_query_with_llm(original_question, num_variations)
                
        except Exception as e:
            logger.error(f"Error in enhanced query expansion: {e}")
            return self.expand_query_with_llm(original_question, num_variations)
    
    def decompose_complex_question(self, question: str, llm=None) -> List[str]:
        """Decompose complex questions into focused sub-questions using enhanced prompts"""
        try:
            if llm is None:
                # Basic decomposition - split by conjunctions
                basic_decomp = []
                if ' and ' in question.lower():
                    parts = question.split(' and ')
                    for part in parts:
                        if part.strip():
                            basic_decomp.append(part.strip())
                elif ' or ' in question.lower():
                    parts = question.split(' or ')
                    for part in parts:
                        if part.strip():
                            basic_decomp.append(part.strip())
                else:
                    basic_decomp = [question]
                
                return basic_decomp[:4]  # Limit to 4 sub-questions
            
            # Use enhanced prompt for question decomposition
            prompt = get_retriever_prompt('question_decomposition_enhanced', question=question)
            response = llm.invoke(prompt)
            
            # Try to extract JSON response
            import json
            try:
                start = response.content.find("[")
                end = response.content.rfind("]") + 1
                sub_questions = json.loads(response.content[start:end])
                
                # Validate sub-questions
                valid_sub_questions = []
                for sub_q in sub_questions:
                    if isinstance(sub_q, str) and sub_q.strip():
                        valid_sub_questions.append(sub_q.strip())
                
                return valid_sub_questions[:4] if valid_sub_questions else [question]
                
            except (json.JSONDecodeError, AttributeError):
                # Fallback to basic decomposition
                logger.warning("Failed to parse LLM response for question decomposition, using fallback")
                return [question]
                
        except Exception as e:
            logger.error(f"Error in question decomposition: {e}")
            return [question]
    
    def generate_enhanced_answer(self, question: str, relevant_chunks: List[str], 
                                processing_route: int = 3, llm=None) -> str:
        """Generate enhanced answers using prompts from api_demo_new.py"""
        try:
            if not relevant_chunks or llm is None:
                return "No relevant information found to answer the question."
            
            # Prepare context from chunks
            if isinstance(relevant_chunks[0], dict):
                # Extract text from chunk objects
                context = "\n\n".join([chunk.get('text', str(chunk)) for chunk in relevant_chunks])
            else:
                # Handle simple string chunks
                context = "\n\n".join(relevant_chunks)
            
            # Use appropriate prompt based on processing route
            if processing_route in [1, 2]:
                prompt = get_retriever_prompt(
                    'document_faithful_strict_enhanced',
                    question=question,
                    document_context=context
                )
            else:
                prompt = get_retriever_prompt(
                    'document_faithful_general_enhanced',
                    question=question,
                    document_context=context
                )
            
            response = llm.invoke(prompt)
            answer = response.content.strip()
            
            # Clean up the answer
            answer = answer.replace('\n', ' ').replace('\\n', ' ')
            answer = answer.replace('\t', ' ').replace('\\t', ' ')
            answer = answer.replace('\r', ' ').replace('\\r', ' ')
            
            # Remove extra whitespace
            import re
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            return answer if answer else "Unable to generate answer from the provided context."
            
        except Exception as e:
            logger.error(f"Error generating enhanced answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def generate_general_knowledge_answer(self, question: str, llm=None) -> str:
        """Generate general knowledge answers using enhanced prompts"""
        try:
            if llm is None:
                return "I need access to a language model to answer general knowledge questions."
            
            prompt = get_retriever_prompt('general_knowledge_enhanced', question=question)
            response = llm.invoke(prompt)
            answer = response.content.strip()
            
            # Clean up the answer similar to api_demo_new.py
            answer = answer.replace('\n', ' ').replace('\\n', ' ').replace('/n', ' ')
            answer = answer.replace('\t', ' ').replace('\\t', ' ').replace('/t', ' ')
            answer = answer.replace('\r', ' ').replace('\\r', ' ').replace('/r', ' ')
            answer = answer.replace('\\', ' ').replace('\"', '').replace("'", "'")
            
            # Remove extra whitespace
            import re
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            return answer if answer else "I'm unable to provide an answer to that question."
            
        except Exception as e:
            logger.error(f"Error generating general knowledge answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def synthesize_complex_answer(self, original_question: str, relevant_chunks: List[str], 
                                 sub_question_results: List[Dict], llm=None) -> str:
        """Synthesize answers from multiple sub-questions using enhanced prompts"""
        try:
            if not sub_question_results or llm is None:
                # Fallback to regular answer generation
                return self.generate_enhanced_answer(original_question, relevant_chunks, llm=llm)
            
            # Prepare synthesis context
            synthesis_context = f"ORIGINAL QUESTION: {original_question}\n\n"
            synthesis_context += "RELEVANT CHUNKS:\n"
            
            for i, chunk in enumerate(relevant_chunks[:3]):  # Limit chunks for brevity
                chunk_text = chunk.get('text', str(chunk)) if isinstance(chunk, dict) else chunk
                synthesis_context += f"Chunk {i+1}: {chunk_text[:300]}...\n\n"
            
            synthesis_context += "SUB-QUESTION RESULTS:\n"
            for i, result in enumerate(sub_question_results):
                if isinstance(result, dict):
                    sub_q = result.get('question', f'Sub-question {i+1}')
                    answer = result.get('answer', 'No answer available')
                else:
                    sub_q = f'Sub-question {i+1}'
                    answer = str(result)
                
                synthesis_context += f"{sub_q}: {answer}\n"
            
            prompt = get_retriever_prompt('complex_synthesis_enhanced', synthesis_context=synthesis_context)
            response = llm.invoke(prompt)
            
            synthesized_answer = response.content.strip()
            
            # Clean up the synthesized answer
            synthesized_answer = synthesized_answer.replace('\n', ' ').replace('\\n', ' ')
            synthesized_answer = synthesized_answer.replace('\t', ' ').replace('\\t', ' ')
            
            # Remove extra whitespace
            import re
            synthesized_answer = re.sub(r'\s+', ' ', synthesized_answer).strip()
            
            return synthesized_answer if synthesized_answer else self.generate_enhanced_answer(original_question, relevant_chunks, llm=llm)
            
        except Exception as e:
            logger.error(f"Error synthesizing complex answer: {e}")
            return self.generate_enhanced_answer(original_question, relevant_chunks, llm=llm)
    
    def detect_puzzle_document(self, document_text: str, sampled_pages: List[str] = None, llm=None) -> Dict[str, Any]:
        """Detect if document contains puzzle/challenge content using enhanced prompts"""
        try:
            if llm is None:
                # Basic heuristic detection
                puzzle_keywords = [
                    'challenge', 'mission', 'api', 'endpoint', 'flight', 'registration',
                    'hackrx', 'puzzle', 'solve', 'task', 'submission', 'deadline'
                ]
                
                text_lower = document_text.lower()
                found_keywords = [kw for kw in puzzle_keywords if kw in text_lower]
                
                is_puzzle = len(found_keywords) >= 2
                confidence = min(1.0, len(found_keywords) / 4)
                
                return {
                    'is_puzzle': is_puzzle,
                    'confidence': confidence,
                    'puzzle_type': 'general_puzzle' if is_puzzle else 'not_puzzle',
                    'key_indicators': found_keywords,
                    'processing_recommendation': 'agent' if is_puzzle else 'standard',
                    'requires_llm_first': is_puzzle
                }
            
            # Use sample of document for analysis
            sample_text = document_text[:1500] if len(document_text) > 1500 else document_text
            if sampled_pages:
                sample_text = "\n\n".join(sampled_pages[:2])[:1500]
            
            prompt = get_retriever_prompt('puzzle_detection_enhanced', document_sample=sample_text)
            response = llm.invoke(prompt)
            
            # Try to extract JSON response
            import json
            try:
                start = response.content.find("{")
                end = response.content.rfind("}") + 1
                detection_result = json.loads(response.content[start:end])
                
                # Validate and set defaults
                detection_result.setdefault('is_puzzle', False)
                detection_result.setdefault('confidence', 0.0)
                detection_result.setdefault('puzzle_type', 'not_puzzle')
                detection_result.setdefault('key_indicators', [])
                detection_result.setdefault('processing_recommendation', 'standard')
                detection_result.setdefault('requires_llm_first', False)
                
                return detection_result
                
            except (json.JSONDecodeError, AttributeError):
                logger.warning("Failed to parse LLM response for puzzle detection, using fallback")
                return {
                    'is_puzzle': False,
                    'confidence': 0.0,
                    'puzzle_type': 'not_puzzle',
                    'key_indicators': [],
                    'processing_recommendation': 'standard',
                    'requires_llm_first': False
                }
                
        except Exception as e:
            logger.error(f"Error in puzzle document detection: {e}")
            return {
                'is_puzzle': False,
                'confidence': 0.0,
                'puzzle_type': 'not_puzzle',
                'key_indicators': [],
                'processing_recommendation': 'standard',
                'requires_llm_first': False
            }
    
    def parallel_semantic_search_enhanced(self, embeddings: np.ndarray, chunks: List[str], 
                                        original_question: str, sub_questions: List[str], 
                                        top_k: int = 5, max_workers: int = 6, 
                                        faiss_index=None) -> Dict[str, Any]:
        """Enhanced parallel search with multiple query variations"""
        try:
            all_results = {}
            
            # Search with original question
            if faiss_index is not None:
                original_results = self.faiss_semantic_search(
                    faiss_index, chunks, original_question, top_k
                )
            else:
                original_results = self.enhanced_semantic_search(
                    embeddings, chunks, original_question, top_k
                )
            
            all_results['original'] = {
                'question': original_question,
                'results': original_results,
                'chunks': [chunks[idx] for idx, _ in original_results if idx < len(chunks)]
            }
            
            # Search with sub-questions in parallel
            if sub_questions:
                def search_sub_question(sub_q):
                    try:
                        if faiss_index is not None:
                            results = self.faiss_semantic_search(
                                faiss_index, chunks, sub_q, top_k
                            )
                        else:
                            results = self.enhanced_semantic_search(
                                embeddings, chunks, sub_q, top_k
                            )
                        return sub_q, results
                    except Exception as e:
                        logger.error(f"Error searching sub-question '{sub_q}': {e}")
                        return sub_q, []
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_question = {
                        executor.submit(search_sub_question, sub_q): sub_q 
                        for sub_q in sub_questions
                    }
                    
                    for future in as_completed(future_to_question):
                        try:
                            sub_q, results = future.result()
                            all_results[sub_q] = {
                                'question': sub_q,
                                'results': results,
                                'chunks': [chunks[idx] for idx, _ in results if idx < len(chunks)]
                            }
                        except Exception as e:
                            logger.error(f"Error processing sub-question result: {e}")
            
            # Aggregate results
            all_chunks = set()
            for question_data in all_results.values():
                all_chunks.update(question_data['chunks'])
            
            return {
                'all_results': all_results,
                'aggregated_chunks': list(all_chunks),
                'total_questions_processed': len(all_results),
                'total_unique_chunks': len(all_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced parallel search: {e}")
            return {'error': str(e)}

# Standalone functions for backward compatibility
def faiss_semantic_search(index, chunks: List[str], question: str, top_k: int = 5) -> List[Tuple[int, float]]:
    """Standalone FAISS semantic search function"""
    retriever = DocumentRetriever()
    return retriever.faiss_semantic_search(index, chunks, question, top_k)

def enhanced_semantic_search(embeddings: Optional[np.ndarray], chunks: List[str], 
                           question: str, top_k: int = 5, faiss_index=None) -> List[Tuple[int, float]]:
    """Standalone enhanced semantic search function"""
    retriever = DocumentRetriever()
    return retriever.enhanced_semantic_search(embeddings, chunks, question, top_k, faiss_index)

def check_relevance_enhanced(question: str, relevant_chunks: List[str]) -> Dict[str, Any]:
    """Standalone enhanced relevance check function"""
    retriever = DocumentRetriever()
    return retriever.check_relevance_enhanced(question, relevant_chunks)

def process_question_with_relevance_check(question: str, embeddings: np.ndarray, 
                                        processed_chunks: List[str], 
                                        faiss_index=None) -> Dict[str, Any]:
    """Standalone question processing function"""
    retriever = DocumentRetriever()
    return retriever.process_question_with_relevance_check(question, embeddings, processed_chunks, faiss_index)

# Enhanced standalone functions using prompts from api_demo_new.py
def expand_query_with_enhanced_llm(original_question: str, num_variations: int = 4, llm=None) -> List[str]:
    """Standalone enhanced query expansion function"""
    retriever = DocumentRetriever()
    return retriever.expand_query_with_enhanced_llm(original_question, num_variations, llm)

def decompose_complex_question(question: str, llm=None) -> List[str]:
    """Standalone question decomposition function"""
    retriever = DocumentRetriever()
    return retriever.decompose_complex_question(question, llm)

def generate_enhanced_answer(question: str, relevant_chunks: List[str], 
                           processing_route: int = 3, llm=None) -> str:
    """Standalone enhanced answer generation function"""
    retriever = DocumentRetriever()
    return retriever.generate_enhanced_answer(question, relevant_chunks, processing_route, llm)

def generate_general_knowledge_answer(question: str, llm=None) -> str:
    """Standalone general knowledge answer function"""
    retriever = DocumentRetriever()
    return retriever.generate_general_knowledge_answer(question, llm)

def synthesize_complex_answer(original_question: str, relevant_chunks: List[str], 
                            sub_question_results: List[Dict], llm=None) -> str:
    """Standalone complex answer synthesis function"""
    retriever = DocumentRetriever()
    return retriever.synthesize_complex_answer(original_question, relevant_chunks, sub_question_results, llm)

def detect_puzzle_document(document_text: str, sampled_pages: List[str] = None, llm=None) -> Dict[str, Any]:
    """Standalone puzzle document detection function"""
    retriever = DocumentRetriever()
    return retriever.detect_puzzle_document(document_text, sampled_pages, llm)

# Alias for backward compatibility
Retriever = DocumentRetriever
