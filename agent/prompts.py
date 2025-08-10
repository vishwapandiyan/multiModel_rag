"""
Agent-specific prompts for challenge solving and puzzle handling.
"""

# Challenge Analysis Prompts
CHALLENGE_ANALYSIS_PROMPT = """
Analyze this document section and extract the challenge structure and rules.

Document Content:
{content_to_analyze}

Please return a JSON response with this exact structure:
{{
  "challenge_type": "description of what type of challenge this is (e.g., flight lookup, car booking, hotel reservation)",
  "initial_api": "the first API endpoint to call to get initial data (just the URL, no GET prefix)",
  "mapping_rules": {{
    "entity_type": "what the first API returns (e.g., city, car, hotel)",
    "target_type": "what it maps to (e.g., landmark, model, location)",
    "mappings": {{
      "entity1": "target1",
      "entity2": "target2"
    }}
  }},
  "endpoint_rules": {{
    "target1": "endpoint1",
    "target2": "endpoint2",
    "default": "default_endpoint"
  }},
  "base_url": "base URL for constructing endpoints",
  "final_result_field": "field name in the final API response that contains the answer"
}}

Extract ALL the rules and mappings mentioned in the document. Be specific about which endpoint to use for which target. For the initial_api, provide ONLY the URL without any HTTP method prefix.
"""

# Entity Mapping Prompts
ENTITY_MAPPING_PROMPT = """
Based on the following challenge context, map the entity '{entity}' to the most appropriate target from the available options.

Available targets: {available_targets}

Consider:
1. Geographic relationships
2. Cultural associations
3. Logical connections

Return only the target name, nothing else.
"""

# Flight Challenge Prompts
FLIGHT_CHALLENGE_DETECTION_PROMPT = """
Analyze this document to determine if it contains a flight challenge puzzle.

Document content:
{document_content}

Look for:
- Flight numbers or aviation terms
- City names and landmarks
- API endpoints related to flights
- Challenge or mission instructions
- Mapping rules between cities and landmarks

Return a JSON response:
{{
  "is_flight_challenge": true|false,
  "confidence": 0.0-1.0,
  "flight_indicators": ["list", "of", "found", "indicators"],
  "cities_found": ["list", "of", "cities"],
  "landmarks_found": ["list", "of", "landmarks"],
  "api_endpoints": ["list", "of", "endpoints"]
}}
"""

# API Challenge Prompts
API_CHALLENGE_DETECTION_PROMPT = """
Analyze this document to determine if it contains an API-based challenge.

Document content:
{document_content}

Look for:
- API endpoints or URLs
- HTTP methods (GET, POST, PUT, DELETE)
- Authentication requirements
- API documentation patterns
- Challenge instructions involving API calls

Return a JSON response:
{{
  "is_api_challenge": true|false,
  "confidence": 0.0-1.0,
  "api_indicators": ["list", "of", "found", "indicators"],
  "endpoints_found": ["list", "of", "endpoints"],
  "methods_found": ["list", "of", "methods"],
  "auth_requirements": ["list", "of", "auth", "requirements"]
}}
"""

# Generic Challenge Reasoning Prompt
CHALLENGE_REASONING_PROMPT = """
You are an intelligent AI assistant helping to solve a challenge. Please analyze the following information carefully and provide a well-reasoned response.

Challenge Context: {challenge_context}

Current Step: {current_step}

Available Information:
{available_information}

Instructions:
- Think step by step
- Consider all relevant information
- Provide clear, actionable responses
- If this involves a puzzle or challenge, break it down systematically

Please provide your analysis and next steps.
"""

def get_agent_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a formatted agent prompt by name with variable substitution.
    
    Args:
        prompt_name: Name of the prompt to retrieve
        **kwargs: Variables to substitute in the prompt
        
    Returns:
        Formatted prompt string
    """
    prompts = {
        'challenge_analysis': CHALLENGE_ANALYSIS_PROMPT,
        'entity_mapping': ENTITY_MAPPING_PROMPT,
        'flight_challenge_detection': FLIGHT_CHALLENGE_DETECTION_PROMPT,
        'api_challenge_detection': API_CHALLENGE_DETECTION_PROMPT,
        'challenge_reasoning': CHALLENGE_REASONING_PROMPT
    }
    
    if prompt_name not in prompts:
        raise ValueError(f"Agent prompt '{prompt_name}' not found. Available prompts: {list(prompts.keys())}")
    
    try:
        return prompts[prompt_name].format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required parameter {e} for agent prompt '{prompt_name}'")
