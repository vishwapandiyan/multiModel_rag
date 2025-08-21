

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
try:
    from document_processing.embedding.embed_generator import EmbeddingGenerator
except ImportError:
    from ..document_processing.embedding.embed_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)

class VectorStore:
    """Main vector store class for managing FAISS indices and vector operations"""
    
    def __init__(self, index_path: str = "vectorstore/faiss_index/index.faiss"):
        self.index_path = index_path
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.embedding_generator = EmbeddingGenerator()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    def create_index(self, embeddings: np.ndarray, index_type: str = "flat") -> bool:
        """
        Create a FAISS index from embeddings
        
        Args:
            embeddings: numpy array of embeddings
            index_type: type of FAISS index ("flat", "ivf", "hnsw")
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if embeddings is None or len(embeddings) == 0:
                logger.error("No embeddings provided for index creation")
                return False
            
            # Normalize embeddings for better performance
            faiss.normalize_L2(embeddings)
            
            # Create appropriate index type
            if index_type == "flat":
                # Exact search, fastest but uses more memory
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
            elif index_type == "ivf":
                # Inverted file index, good balance of speed and memory
                nlist = min(100, max(1, embeddings.shape[0] // 10))
                quantizer = faiss.IndexFlatIP(embeddings.shape[1])
                self.index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], nlist)
                # Train the index
                self.index.train(embeddings)
            elif index_type == "hnsw":
                # Hierarchical Navigable Small World, good for large datasets
                self.index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
            else:
                logger.warning(f"Unknown index type {index_type}, using flat index")
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
            
            # Add vectors to index
            self.index.add(embeddings)
            self.embeddings = embeddings
            
            logger.info(f"Created {index_type} FAISS index with {len(embeddings)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return False
    
    def add_vectors(self, new_embeddings: np.ndarray) -> bool:
        """
        Add new vectors to existing index
        
        Args:
            new_embeddings: numpy array of new embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.index is None:
                logger.error("No index exists, create one first")
                return False
            
            if new_embeddings is None or len(new_embeddings) == 0:
                logger.error("No new embeddings provided")
                return False
            
            # Normalize new embeddings
            faiss.normalize_L2(new_embeddings)
            
            # Add to index
            self.index.add(new_embeddings)
            
            # Update stored embeddings
            if self.embeddings is not None:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                self.embeddings = new_embeddings
            
            logger.info(f"Added {len(new_embeddings)} new vectors to index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors to index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: query vector
            top_k: number of top results to return
            
        Returns:
            tuple: (distances, indices) arrays
        """
        try:
            if self.index is None:
                logger.error("No index exists for search")
                return np.array([]), np.array([])
            
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            distances, indices = self.index.search(query_embedding, top_k)
            
            logger.info(f"Search returned {len(indices[0])} results")
            return distances[0], indices[0]
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return np.array([]), np.array([])
    
    def save_index(self, index_path: Optional[str] = None) -> bool:
        """
        Save FAISS index to disk
        
        Args:
            index_path: path to save index (uses default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.index is None:
                logger.error("No index to save")
                return False
            
            save_path = index_path or self.index_path
            
            # Save FAISS index
            faiss.write_index(self.index, save_path)
            
            # Save metadata (chunks, embeddings info)
            metadata_path = save_path.replace('.faiss', '_metadata.pkl')
            metadata = {
                'chunks': self.chunks,
                'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
                'index_type': type(self.index).__name__
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved FAISS index to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
            return False
    
    def load_index(self, index_path: Optional[str] = None) -> bool:
        """
        Load FAISS index from disk
        
        Args:
            index_path: path to load index from (uses default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            load_path = index_path or self.index_path
            
            if not os.path.exists(load_path):
                logger.error(f"Index file not found: {load_path}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(load_path)
            
            # Load metadata
            metadata_path = load_path.replace('.faiss', '_metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.chunks = metadata.get('chunks', [])
                embeddings_shape = metadata.get('embeddings_shape')
                
                if embeddings_shape:
                    logger.info(f"Loaded index with {embeddings_shape[0]} vectors")
                else:
                    logger.warning("No embeddings shape information in metadata")
            else:
                logger.warning("No metadata file found, chunks not loaded")
            
            logger.info(f"Loaded FAISS index from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index"""
        if self.index is None:
            return {'error': 'No index loaded'}
        
        try:
            info = {
                'index_type': type(self.index).__name__,
                'total_vectors': self.index.ntotal,
                'vector_dimension': self.index.d,
                'is_trained': getattr(self.index, 'is_trained', True),
                'chunks_count': len(self.chunks),
                'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
            }
            
            # Add index-specific information
            if hasattr(self.index, 'nlist'):
                info['nlist'] = self.index.nlist
            
            if hasattr(self.index, 'hnsw'):
                info['hnsw_ef_construction'] = self.index.hnsw.efConstruction
                info['hnsw_ef_search'] = self.index.hnsw.efSearch
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {'error': str(e)}
    
    def update_chunks(self, new_chunks: List[str]) -> bool:
        """
        Update the chunks associated with the index
        
        Args:
            new_chunks: list of new text chunks
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.chunks = new_chunks
            logger.info(f"Updated chunks, total count: {len(self.chunks)}")
            return True
        except Exception as e:
            logger.error(f"Error updating chunks: {e}")
            return False
    
    def remove_vectors(self, indices: List[int]) -> bool:
        """
        Remove vectors from the index (if supported)
        
        Args:
            indices: list of indices to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.index is None:
                logger.error("No index to remove vectors from")
                return False
            
            # Check if index supports removal
            if not hasattr(self.index, 'remove_ids'):
                logger.warning("Current index type doesn't support vector removal")
                return False
            
            # Convert to int64 array as required by FAISS
            indices_array = np.array(indices, dtype=np.int64)
            
            # Remove vectors
            self.index.remove_ids(indices_array)
            
            # Update stored embeddings
            if self.embeddings is not None:
                mask = np.ones(len(self.embeddings), dtype=bool)
                mask[indices] = False
                self.embeddings = self.embeddings[mask]
            
            # Update chunks
            self.chunks = [chunk for i, chunk in enumerate(self.chunks) if i not in indices]
            
            logger.info(f"Removed {len(indices)} vectors from index")
            return True
            
        except Exception as e:
            logger.error(f"Error removing vectors: {e}")
            return False
    
    def optimize_index(self) -> bool:
        """
        Optimize the index for better performance
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.index is None:
                logger.error("No index to optimize")
                return False
            
            # For IVF indices, optimize by training
            if hasattr(self.index, 'train') and not self.index.is_trained:
                if self.embeddings is not None:
                    self.index.train(self.embeddings)
                    logger.info("Trained IVF index")
            
            # For HNSW indices, adjust search parameters
            if hasattr(self.index, 'hnsw'):
                # Optimize search parameters based on dataset size
                if self.index.ntotal > 10000:
                    self.index.hnsw.efSearch = 100
                else:
                    self.index.hnsw.efSearch = 50
                logger.info("Optimized HNSW search parameters")
            
            logger.info("Index optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            return False

# Standalone functions for backward compatibility
def create_faiss_index(embeddings: np.ndarray, index_type: str = "flat") -> Optional[faiss.Index]:
    """Create a FAISS index from embeddings"""
    try:
        vector_store = VectorStore()
        if vector_store.create_index(embeddings, index_type):
            return vector_store.index
        return None
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return None

def save_faiss_index(index: faiss.Index, index_path: str) -> bool:
    """Save a FAISS index to disk"""
    try:
        vector_store = VectorStore(index_path)
        vector_store.index = index
        return vector_store.save_index(index_path)
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        return False

def load_faiss_index(index_path: str) -> Optional[faiss.Index]:
    """Load a FAISS index from disk"""
    try:
        vector_store = VectorStore(index_path)
        if vector_store.load_index(index_path):
            return vector_store.index
        return None
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None
