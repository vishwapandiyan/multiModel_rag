

# Document Route Analysis Prompts (from api_demo_new.py)
DOCUMENT_ROUTE_ANALYSIS_PROMPT = """
Analyze this document sample and determine if it contains structured metadata and what type of document it is.

Document Sample:
{combined_text}

Return ONLY valid JSON:
{{
    "is_structured": true/false,
    "confidence": 0.0-1.0,
    "contains_metadata": true/false,
    "document_type": "legal|financial|technical|academic|other",
    "reasoning": "brief explanation"
}}
"""

# Chunk Processing Prompts (from api_demo_new.py)
CHUNK_ANALYSIS_ENHANCED_PROMPT = """
Analyze these document chunks and extract structured information for each:

{batch_text}

Return ONLY valid JSON array with 15 items (one for each chunk):
[
    {{
        "chunk_id": "chunk_X",
        "text": "exact text content",
        "main_intent": "General|Procedures|Requirements|Definitions|Guidelines",
        "sub_intents": ["specific topics mentioned"],
        "metadata": {{
            "section": "section name",
            "clause_number": "clause number if available",
            "amount": "amount if mentioned",
            "time_period": "time period if mentioned"
        }}
    }}
]
"""

# General Knowledge Answer Prompts (from api_demo_new.py)
GENERAL_KNOWLEDGE_ENHANCED_PROMPT = """You are a helpful assistant. Answer this question directly and briefly.

Question: "{question}"

CRITICAL RULES - FOLLOW EXACTLY:
1. Answer in 1-2 sentences maximum
2. Be direct and helpful
3. NEVER mention documents, policies, or document processing
4. NEVER say "unfortunately", "the document doesn't contain", "not found in document", "unrelated to document", "provided document", or similar phrases
5. NEVER mention being a document analyst or relying on context
6. Answer as if this is a direct question to you personally
7. For programming questions, provide working code
8. For general knowledge, provide the answer directly
9. Use simple, human-like language without technical terms

Answer:"""

# Complex Question Synthesis Prompts (from api_demo_new.py)
COMPLEX_SYNTHESIS_ENHANCED_PROMPT = """ROLE: Expert Document Analyst specializing in concise, clear communication.

CONTEXT: You are answering a complex question that was broken down into multiple sub-questions. Synthesize information from all sub-questions to provide a clear, concise answer.

{synthesis_context}

SYNTHESIS REQUIREMENTS:
1. BE CONCISE: Maximum 2-3 sentences (200-400 characters total)
2. DIRECT ANSWER: Start immediately with the key findings
3. INTEGRATE KEY POINTS: Combine the most important information seamlessly
4. CLEAR INTENT: Make the answer's purpose immediately obvious
5. NO TECHNICAL TERMS: Never mention "sub-questions", "data set", "synthesis", or internal processing details
6. CLEAN REFERENCES: Use only natural document references like "According to the document" or specific clause numbers

ANSWER STRUCTURE:
- Lead with the direct answer to the main question
- Add 2-3 key supporting points maximum
- Include only the most critical references
- End cleanly without unnecessary guidance

EXAMPLE GOOD SYNTHESIS:
"Pincode 600001 has the highest average salary of ₹82,000. Key contributors include Rohit Mehra, Swati Saxena, and Neha Verma."

EXAMPLE BAD SYNTHESIS (technical terms):
"Based on Sub-question 1 analysis of the data set, pincode 600001 shows the highest average salary according to the synthesis of multiple sub-questions..."

CRITICAL: Never use terms like "Sub-question", "data set", "synthesis", "analysis", or reference internal processing steps.

Answer the original question concisely and clearly:"""

# Document Faithful Answer Prompts (from api_demo_new.py)
DOCUMENT_FAITHFUL_STRICT_ENHANCED_PROMPT = """ROLE: Document Reader - You MUST provide information EXACTLY as stated in the document.

CRITICAL INSTRUCTION: Report EXACTLY what the document says, even if it contains errors, test information, or incorrect calculations. DO NOT correct or verify the information.

USER QUESTION: "{question}"

DOCUMENT CONTEXT:
{document_context}

RESPONSE REQUIREMENTS:
1. Answer ONLY based on what is written in the document
2. Use EXACT numbers, dates, calculations from document (even if wrong)
3. Use EXACT terminology from document (even if incorrect)
4. If document contains errors or test data, report it as-is
5. Keep response under 200 characters
6. DO NOT add corrections, verifications, or external knowledge
7. FORMAT: Single paragraph, no bullets, no line breaks

EXAMPLE RESPONSES:
- If document says "Article 17 abolishes privy purse" → Report exactly that
- If document says "2+2=5" → Report exactly that  
- If document contains test calculations → Report exactly as written

Answer from document:"""

DOCUMENT_FAITHFUL_GENERAL_ENHANCED_PROMPT = """ROLE: Document Reader - You MUST provide information EXACTLY as stated in the document.

CRITICAL INSTRUCTION: Report EXACTLY what the document says, even if it contains errors, test information, or incorrect calculations. DO NOT correct or verify the information.

USER QUESTION: "{question}"

DOCUMENT CONTEXT:
{document_context}

RESPONSE REQUIREMENTS:
1. Answer ONLY based on what is written in the document
2. Use EXACT numbers, dates, calculations from document (even if wrong)
3. Use EXACT terminology from document (even if incorrect)  
4. If document contains errors or test data, report it as-is
5. Keep response under 300 characters
6. DO NOT add corrections, verifications, or external knowledge
7. FORMAT: Single paragraph, no bullets, no line breaks

Answer from document:"""

# URL-based Answer Prompts (from api_demo_new.py)
URL_BASED_ANSWER_ENHANCED_PROMPT = """Based on the URL and filename provided, answer this question about the document:

Question: "{question}"

URL Context:
{url_context}

INSTRUCTIONS:
1. Analyze the filename, path, and domain to infer what this document might contain
2. Provide a helpful answer based on the URL structure and filename
3. Be direct and specific based on available URL information
4. If the filename suggests a specific type of content, mention it
5. Keep response concise (2-3 sentences maximum)
6. Do not mention that the file cannot be processed - focus on what it likely contains

Answer:"""

ZIP_URL_ANSWER_ENHANCED_PROMPT = """Based on the URL and ZIP filename provided, answer this question intelligently:

Question: "{question}"

URL Context:
{url_context}

INSTRUCTIONS FOR ZIP FILES:
1. Analyze the filename, path, and domain to infer what this ZIP archive might contain
2. Consider the naming patterns, domain context, and path structure
3. ZIP files typically contain collections of related files (documents, software, data, etc.)
4. Provide a helpful answer based on the URL analysis and common ZIP file patterns
5. If the question is about content, explain what the ZIP likely contains based on URL clues
6. If the question is unrelated to the ZIP, answer it directly using general knowledge
7. Keep response concise and informative
8. Do not mention processing limitations - focus on providing useful information

Answer:"""

# Question Decomposition Prompts (from api_demo_new.py)
QUESTION_DECOMPOSITION_ENHANCED_PROMPT = """
Analyze this complex question and break it down into 2-4 focused sub-questions. Each sub-question should:
1. Address a specific aspect of the original question
2. Be self-contained and answerable independently
3. Cover all aspects of the original question when combined
4. Be clear and specific

Original Question: "{question}"

Return ONLY a JSON array with 2-4 sub-questions:
[
    "sub-question 1",
    "sub-question 2", 
    "sub-question 3",
    "sub-question 4"
]

Focus on clarity and comprehensive coverage of the original question.
"""

# Query Expansion Prompts (from api_demo_new.py)
QUERY_EXPANSION_ENHANCED_PROMPT = """
Generate {num_variations} different ways to ask the same question. Each variation should:
1. Use different words but mean the same thing
2. Include synonyms and related terms
3. Maintain the same intent and scope
4. Be suitable for document search

Original Question: "{original_question}"

Return ONLY a JSON array with {num_variations} question variations:
["variation 1", "variation 2", "variation 3", "variation 4"]

Focus on semantic similarity while using diverse vocabulary.
"""

# Malayalam Document Prompts (from api_demo_new.py)
MALAYALAM_DOCUMENT_ENGLISH_ENHANCED_PROMPT = """You are a friendly and polite assistant. Please help me understand this information from the Malayalam document in a warm, conversational way.

Document (Malayalam): {predefined_content}

Question: {question}

Please respond in a polite, human-like manner using simple English. Be warm and helpful, like talking to a friend. Keep your answer between 100-300 characters and make it easy to understand:"""

MALAYALAM_DOCUMENT_MALAYALAM_ENHANCED_PROMPT = """You are a friendly and polite assistant. Please help me by answering this question in a warm, conversational way in Malayalam.

Document (Malayalam): {predefined_content}

Question: {question}

Please respond in a polite, human-like manner using simple Malayalam. Be warm and helpful, like talking to a friend. Keep your answer between 100-300 characters and make it easy to understand:"""

# Relevance Check Prompts
RELEVANCE_CHECK_ENHANCED_PROMPT = """
Analyze whether the following question is relevant to the provided document chunks and determine the confidence level.

Question: "{question}"

Document Chunks:
{chunks_text}

Evaluate:
1. Semantic similarity between question and content
2. Presence of relevant keywords and concepts
3. Overall answerable potential from the document
4. Confidence in the relevance assessment

Return a JSON response:
{{
    "is_relevant": true|false,
    "relevance_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation of relevance",
    "key_matches": ["list", "of", "matching", "concepts"],
    "suggested_approach": "document_based|general_knowledge|hybrid"
}}
"""

# Enhanced Answer Generation Prompts
ENHANCED_ANSWER_GENERATION_PROMPT = """Based on the provided document chunks, answer the question comprehensively using semantic understanding.

Question: "{question}"

Relevant Document Context:
{context}

Processing Route: {processing_route}

Instructions:
- Prioritize information from the provided context
- Use semantic understanding to connect related concepts
- Provide comprehensive but concise answers
- Structure your response clearly and logically
- If context is insufficient, clearly indicate limitations
- Maintain accuracy and relevance
- Maximum 250 words

Answer:"""

# Puzzle Document Detection Prompts (from api_demo_new.py)
PUZZLE_DETECTION_ENHANCED_PROMPT = """
Analyze this document sample to determine if it contains puzzle, challenge, or mission-type content that requires special agent processing.

Document Sample:
{document_sample}

Look for specific indicators such as:
- Challenge or mission instructions
- API endpoints or technical tasks
- Step-by-step procedures requiring execution
- Flight numbers, cities, or specific landmarks
- Registration or submission requirements
- Hackathon or competition elements
- Time-sensitive tasks or deadlines

Return a JSON response:
{{
    "is_puzzle": true|false,
    "confidence": 0.0-1.0,
    "puzzle_type": "flight_challenge|api_challenge|hackrx_puzzle|general_puzzle|not_puzzle",
    "key_indicators": ["list", "of", "found", "indicators"],
    "processing_recommendation": "agent|standard",
    "requires_llm_first": true|false,
    "reasoning": "explanation for the classification"
}}
"""

# Function to get prompts by category
def get_retriever_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a formatted prompt by name with variable substitution for retriever module.
    
    Args:
        prompt_name: Name of the prompt to retrieve
        **kwargs: Variables to substitute in the prompt
        
    Returns:
        Formatted prompt string
    """
    prompts = {
        'document_route_analysis': DOCUMENT_ROUTE_ANALYSIS_PROMPT,
        'chunk_analysis_enhanced': CHUNK_ANALYSIS_ENHANCED_PROMPT,
        'general_knowledge_enhanced': GENERAL_KNOWLEDGE_ENHANCED_PROMPT,
        'complex_synthesis_enhanced': COMPLEX_SYNTHESIS_ENHANCED_PROMPT,
        'document_faithful_strict_enhanced': DOCUMENT_FAITHFUL_STRICT_ENHANCED_PROMPT,
        'document_faithful_general_enhanced': DOCUMENT_FAITHFUL_GENERAL_ENHANCED_PROMPT,
        'url_based_answer_enhanced': URL_BASED_ANSWER_ENHANCED_PROMPT,
        'zip_url_answer_enhanced': ZIP_URL_ANSWER_ENHANCED_PROMPT,
        'question_decomposition_enhanced': QUESTION_DECOMPOSITION_ENHANCED_PROMPT,
        'query_expansion_enhanced': QUERY_EXPANSION_ENHANCED_PROMPT,
        'malayalam_english_enhanced': MALAYALAM_DOCUMENT_ENGLISH_ENHANCED_PROMPT,
        'malayalam_malayalam_enhanced': MALAYALAM_DOCUMENT_MALAYALAM_ENHANCED_PROMPT,
        'relevance_check_enhanced': RELEVANCE_CHECK_ENHANCED_PROMPT,
        'enhanced_answer_generation': ENHANCED_ANSWER_GENERATION_PROMPT,
        'puzzle_detection_enhanced': PUZZLE_DETECTION_ENHANCED_PROMPT
    }
    
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(prompts.keys())}")
    
    try:
        return prompts[prompt_name].format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required parameter {e} for prompt '{prompt_name}'")

# Prompt categories for easy reference
RETRIEVER_PROMPT_CATEGORIES = {
    'document_analysis': [
        'document_route_analysis',
        'chunk_analysis_enhanced',
        'puzzle_detection_enhanced'
    ],
    'question_processing': [
        'question_decomposition_enhanced',
        'query_expansion_enhanced',
        'relevance_check_enhanced'
    ],
    'answer_generation': [
        'general_knowledge_enhanced',
        'complex_synthesis_enhanced',
        'document_faithful_strict_enhanced',
        'document_faithful_general_enhanced',
        'enhanced_answer_generation'
    ],
    'specialized': [
        'url_based_answer_enhanced',
        'zip_url_answer_enhanced',
        'malayalam_english_enhanced',
        'malayalam_malayalam_enhanced'
    ]
}

def get_prompts_by_category(category: str) -> list:
    """Get all prompt names in a specific category."""
    return RETRIEVER_PROMPT_CATEGORIES.get(category, [])

def list_all_retriever_prompts() -> list:
    """Get a list of all available prompt names in the retriever."""
    return [
        'document_route_analysis', 'chunk_analysis_enhanced', 'general_knowledge_enhanced',
        'complex_synthesis_enhanced', 'document_faithful_strict_enhanced', 
        'document_faithful_general_enhanced', 'url_based_answer_enhanced', 
        'zip_url_answer_enhanced', 'question_decomposition_enhanced',
        'query_expansion_enhanced', 'malayalam_english_enhanced', 
        'malayalam_malayalam_enhanced', 'relevance_check_enhanced',
        'enhanced_answer_generation', 'puzzle_detection_enhanced'
    ]
