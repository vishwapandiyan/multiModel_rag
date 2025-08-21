

# Document Route Analysis Prompts
DOCUMENT_ROUTE_ANALYSIS_PROMPT = """
Analyze this document sample and determine if it contains structured metadata and what type of document it is.

Document Sample:
{document_sample}

Return a JSON response with this exact structure:
{{
    "document_type": "legal_contract|technical_spec|general_document|structured_data",
    "complexity_level": "low|medium|high",
    "contains_structured_data": true|false,
    "recommended_route": 1|2|3,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

# Chunk Processing Prompts
CHUNK_ANALYSIS_PROMPT = """
Analyze these document chunks and extract structured information for each:

{batch_text}

Return a JSON array with this exact structure for each chunk:
[
    {{
        "chunk_id": "chunk_0",
        "main_intent": "primary purpose of this chunk",
        "sub_intents": ["list", "of", "secondary", "purposes"],
        "metadata": {{
            "section": "section name",
            "clause_number": "if applicable",
            "importance": "high|medium|low"
        }}
    }}
]
"""

# General Knowledge Answer Prompts
GENERAL_KNOWLEDGE_PROMPT = """You are a helpful assistant. Answer this question directly and briefly.

Question: "{question}"

Instructions:
- Provide a clear, concise answer
- Use simple language
- Be factual and accurate
- Keep the response under 100 words
"""

# Complex Question Synthesis Prompts
COMPLEX_SYNTHESIS_PROMPT = """ROLE: Expert Document Analyst specializing in concise, clear communication.

CONTEXT: You are answering a complex question that was broken down into multiple sub-questions. Synthesize information from all sub-questions to provide a clear, concise answer.

ORIGINAL QUESTION: {original_question}

RELEVANT CHUNKS:
{relevant_chunks}

SUB-QUESTION RESULTS:
{sub_question_results}

TASK: Synthesize all information into ONE clear, comprehensive answer that directly addresses the original question.

REQUIREMENTS:
- Be concise but complete
- Prioritize the most relevant information
- Avoid repetition
- Use clear, professional language
- Maximum 150 words
"""

# Document Faithful Answer Prompts
DOCUMENT_FAITHFUL_STRICT_PROMPT = """ROLE: Document Reader - You MUST provide information EXACTLY as stated in the document.

CRITICAL INSTRUCTION: Report EXACTLY what the document says, even if it contains errors, test information, or incorrect calculations. DO NOT correct or verify the information.

QUESTION: {question}

DOCUMENT CONTEXT:
{document_context}

REQUIREMENTS:
- Quote directly from the document when possible
- If information is not in the document, say "This information is not provided in the document"
- DO NOT add external knowledge or corrections
- BE COMPLETELY FAITHFUL to the source material
- Use phrases like "According to the document" or "The document states"
"""

DOCUMENT_FAITHFUL_GENERAL_PROMPT = """ROLE: Document Reader - You MUST provide information EXACTLY as stated in the document.

CRITICAL INSTRUCTION: Report EXACTLY what the document says, even if it contains errors, test information, or incorrect calculations. DO NOT correct or verify the information.

QUESTION: {question}

DOCUMENT CONTEXT:
{document_context}

TASK: Answer the question using ONLY information from the provided document context.

REQUIREMENTS:
- Base your answer ONLY on the provided document
- Quote directly when relevant
- If the document doesn't contain the answer, clearly state this
- Maintain the document's perspective and terminology
- Maximum 100 words
"""

# URL-based Answer Prompts
URL_BASED_ANSWER_PROMPT = """Based on the URL and filename provided, answer this question about the document:

Question: "{question}"

URL: {url}
Filename: {filename}

Instructions:
- Analyze the URL structure and filename for clues
- Make reasonable inferences about the document content
- Be helpful while acknowledging limitations
- Keep response under 80 words
"""

ZIP_URL_ANSWER_PROMPT = """Based on the URL and ZIP filename provided, answer this question intelligently:

Question: "{question}"

URL: {url}
ZIP Filename: {zip_filename}

Instructions:
- Consider what the ZIP file might contain based on its name and URL
- Make educated guesses about the content
- Acknowledge that this is an inference
- Keep response under 80 words
"""

# Question Decomposition Prompts
QUESTION_DECOMPOSITION_PROMPT = """
Analyze this complex question and break it down into 2-4 focused sub-questions. Each sub-question should:
1. Address a specific aspect of the original question
2. Be self-contained and answerable independently
3. Cover all important parts of the original question
4. Use clear, simple language

Original Question: "{question}"

Return a JSON array of sub-questions:
["sub-question 1", "sub-question 2", "sub-question 3"]

Focus on clarity and comprehensive coverage of the original question.
"""

# Query Expansion Prompts
QUERY_EXPANSION_PROMPT = """
Generate {num_variations} different ways to ask the same question. Each variation should:
1. Use different words but mean the same thing
2. Include synonyms and related terms
3. Maintain the same intent and scope
4. Be suitable for document search

Original Question: "{original_question}"

Return a JSON array of question variations:
["variation 1", "variation 2", "variation 3", "variation 4"]

Focus on semantic similarity while using diverse vocabulary.
"""

# Malayalam Document Prompts
MALAYALAM_DOCUMENT_ENGLISH_PROMPT = """You are a friendly and polite assistant. Please help me understand this information from the Malayalam document in a warm, conversational way.

Document (Malayalam): {predefined_content}

Question: {question}

Please provide a helpful response in English, being warm and conversational while staying accurate to the document content.
"""

MALAYALAM_DOCUMENT_MALAYALAM_PROMPT = """You are a friendly and polite assistant. Please help me by answering this question in a warm, conversational way in Malayalam.

Document (Malayalam): {predefined_content}

Question (Malayalam): {question}

Please provide a helpful response in Malayalam, being warm and conversational while staying accurate to the document content.
"""

# Relevance Check Prompts
RELEVANCE_CHECK_PROMPT = """
Analyze whether the following question is relevant to the provided document chunks.

Question: "{question}"

Document Chunks:
{chunks_text}

Determine:
1. Is this question answerable from the document content?
2. How relevant is the question to the document (0-100%)?
3. What specific parts of the document relate to the question?

Return a JSON response:
{{
    "is_relevant": true|false,
    "relevance_score": 0-100,
    "reasoning": "explanation of relevance",
    "related_content": ["list", "of", "relevant", "parts"]
}}
"""

# Enhanced Answer Generation Prompts
ENHANCED_ANSWER_PROMPT = """Based on the provided document chunks, answer the question comprehensively and accurately.

Question: "{question}"

Relevant Document Context:
{context}

Instructions:
- Use information primarily from the provided context
- Be comprehensive but concise
- Structure your answer clearly
- If the context doesn't fully answer the question, indicate what's missing
- Maximum 200 words
"""

# Puzzle Document Detection Prompts
PUZZLE_DETECTION_PROMPT = """
Analyze this document sample to determine if it contains puzzle, challenge, or mission-type content.

Document Sample:
{document_sample}

Look for indicators such as:
- Challenge or mission instructions
- API endpoints or technical tasks
- Step-by-step procedures
- Flight numbers, cities, or landmarks
- Registration or submission requirements

Return a JSON response:
{{
    "is_puzzle": true|false,
    "confidence": 0.0-1.0,
    "puzzle_type": "flight_challenge|api_challenge|general_puzzle|not_puzzle",
    "key_indicators": ["list", "of", "found", "indicators"],
    "processing_recommendation": "agent|standard"
}}
"""

# Function to get prompts by category
def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a formatted prompt by name with variable substitution.
    
    Args:
        prompt_name: Name of the prompt to retrieve
        **kwargs: Variables to substitute in the prompt
        
    Returns:
        Formatted prompt string
    """
    prompts = {
        'document_route_analysis': DOCUMENT_ROUTE_ANALYSIS_PROMPT,
        'chunk_analysis': CHUNK_ANALYSIS_PROMPT,
        'general_knowledge': GENERAL_KNOWLEDGE_PROMPT,
        'complex_synthesis': COMPLEX_SYNTHESIS_PROMPT,
        'document_faithful_strict': DOCUMENT_FAITHFUL_STRICT_PROMPT,
        'document_faithful_general': DOCUMENT_FAITHFUL_GENERAL_PROMPT,
        'url_based_answer': URL_BASED_ANSWER_PROMPT,
        'zip_url_answer': ZIP_URL_ANSWER_PROMPT,
        'question_decomposition': QUESTION_DECOMPOSITION_PROMPT,
        'query_expansion': QUERY_EXPANSION_PROMPT,
        'malayalam_english': MALAYALAM_DOCUMENT_ENGLISH_PROMPT,
        'malayalam_malayalam': MALAYALAM_DOCUMENT_MALAYALAM_PROMPT,
        'relevance_check': RELEVANCE_CHECK_PROMPT,
        'enhanced_answer': ENHANCED_ANSWER_PROMPT,
        'puzzle_detection': PUZZLE_DETECTION_PROMPT
    }
    
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(prompts.keys())}")
    
    try:
        return prompts[prompt_name].format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required parameter {e} for prompt '{prompt_name}'")

# Prompt categories for easy reference
PROMPT_CATEGORIES = {
    'document_analysis': [
        'document_route_analysis',
        'chunk_analysis',
        'puzzle_detection'
    ],
    'question_processing': [
        'question_decomposition',
        'query_expansion',
        'relevance_check'
    ],
    'answer_generation': [
        'general_knowledge',
        'complex_synthesis',
        'document_faithful_strict',
        'document_faithful_general',
        'enhanced_answer'
    ],
    'specialized': [
        'url_based_answer',
        'zip_url_answer',
        'malayalam_english',
        'malayalam_malayalam'
    ]
}

def get_prompts_by_category(category: str) -> list:
    """Get all prompt names in a specific category."""
    return PROMPT_CATEGORIES.get(category, [])

def list_all_prompts() -> list:
    """Get a list of all available prompt names."""
    return [
        'document_route_analysis', 'chunk_analysis', 'general_knowledge',
        'complex_synthesis', 'document_faithful_strict', 'document_faithful_general',
        'url_based_answer', 'zip_url_answer', 'question_decomposition',
        'query_expansion', 'malayalam_english', 'malayalam_malayalam',
        'relevance_check', 'enhanced_answer', 'puzzle_detection'
    ]
