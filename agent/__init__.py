"""
Agent Module

Handles agent-based puzzle/challenge solving using LangGraph.
"""

from .new_agent_simple import (
    GenericChallengeHandler,
    SimpleWorkflowAgent,
    LangGraphGenericAgent
)

__all__ = [
    'GenericChallengeHandler',
    'SimpleWorkflowAgent', 
    'LangGraphGenericAgent'
]
