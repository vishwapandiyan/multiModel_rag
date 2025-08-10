# Enhanced Retriever Module

The retriever module has been enhanced with prompts and functionality extracted from `api_demo_new.py` to provide better document retrieval and answer generation capabilities.

## New Features Added

### 1. Enhanced Prompts Module (`prompts.py`)
All prompts from `api_demo_new.py` have been extracted and organized into a dedicated prompts module:

- **Document Analysis Prompts**: Route analysis, chunk processing, puzzle detection
- **Question Processing Prompts**: Question decomposition, query expansion, relevance checking  
- **Answer Generation Prompts**: General knowledge, document-faithful answers, complex synthesis
- **Specialized Prompts**: URL-based answers, Malayalam document handling

### 2. Enhanced DocumentRetriever Class
New methods added to the `DocumentRetriever` class:

#### Query Processing
- `expand_query_with_enhanced_llm()`: Enhanced query expansion using LLM prompts
- `decompose_complex_question()`: Break complex questions into sub-questions
- `detect_puzzle_document()`: Detect puzzle/challenge documents that need special processing

#### Answer Generation  
- `generate_enhanced_answer()`: Generate answers using document-faithful prompts
- `generate_general_knowledge_answer()`: Generate general knowledge answers
- `synthesize_complex_answer()`: Synthesize answers from multiple sub-questions

### 3. Standalone Functions
For backward compatibility with `api_demo_new.py`, standalone functions are provided:

```python
from hackrx_final_x2X.retriever.retriever import (
    expand_query_with_enhanced_llm,
    decompose_complex_question, 
    generate_enhanced_answer,
    generate_general_knowledge_answer,
    synthesize_complex_answer,
    detect_puzzle_document
)
```

## Usage Examples

### Basic Usage
```python
from hackrx_final_x2X.retriever.retriever import DocumentRetriever

retriever = DocumentRetriever()

# Enhanced query expansion
variations = retriever.expand_query_with_enhanced_llm(
    "What is the salary policy?", 
    num_variations=3, 
    llm=llm_instance
)

# Question decomposition
sub_questions = retriever.decompose_complex_question(
    "What are the salary ranges and benefits for engineers?",
    llm=llm_instance
)

# Enhanced answer generation
answer = retriever.generate_enhanced_answer(
    question="What is the leave policy?",
    relevant_chunks=retrieved_chunks,
    processing_route=2,
    llm=llm_instance
)
```

### Puzzle Document Detection
```python
# Detect if document contains puzzles/challenges
puzzle_result = retriever.detect_puzzle_document(
    document_text=document_content,
    sampled_pages=sample_pages,
    llm=llm_instance
)

if puzzle_result['is_puzzle'] and puzzle_result['requires_llm_first']:
    # Process through agent pipeline first
    print("Document requires LLM processing before standard pipeline")
```

## Integration with Existing Pipeline

The enhanced retriever maintains full backward compatibility while adding new capabilities:

1. **Existing code** continues to work without changes
2. **New features** can be used by passing an `llm` parameter
3. **Fallback behavior** ensures graceful degradation when LLM is unavailable
4. **Memory configuration** follows the existing pattern [[memory:5746337]] for puzzle documents

## Prompt Categories

Prompts are organized into logical categories:

- `document_analysis`: Route analysis, chunk processing, puzzle detection
- `question_processing`: Decomposition, expansion, relevance checking
- `answer_generation`: Various answer generation strategies
- `specialized`: Language-specific and format-specific prompts

Access prompts using:
```python
from hackrx_final_x2X.retriever.prompts import get_retriever_prompt

prompt = get_retriever_prompt('general_knowledge_enhanced', question="What is AI?")
```
