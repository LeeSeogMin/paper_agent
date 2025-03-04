"""
User Interaction Agent Module
Manages direct interactions with users, including input collection, result presentation,
and conversation management throughout the paper writing workflow.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI

from config.settings import OUTPUT_DIR
from utils.logger import logger
from models.paper import Paper
from models.state import PaperWorkflowState
from agents.base import BaseAgent
from langchain_community.chat_models import ChatXAI


class UserQuery(BaseModel):
    """User query format"""
    query_text: str = Field(description="User's original query text")
    timestamp: str = Field(description="Query timestamp")
    query_type: str = Field(description="Type of query (question, command, feedback)")
    context: Dict[str, Any] = Field(description="Context information", default_factory=dict)


class QueryAnalysis(BaseModel):
    """Query analysis result format"""
    intent: str = Field(description="Identified user intent")
    entities: List[str] = Field(description="Extracted entities from query")
    required_actions: List[str] = Field(description="Actions needed to fulfill query")
    confidence_score: float = Field(description="Confidence in analysis (0.0-1.0)")
    suggested_response: str = Field(description="Suggested response template")


class UserPreferences(BaseModel):
    """User preferences format"""
    language: str = Field(description="Preferred language", default="en")
    output_format: str = Field(description="Preferred output format", default="markdown")
    notification_level: str = Field(description="Notification verbosity", default="standard")
    style_preferences: Dict[str, Any] = Field(description="Style preferences", default_factory=dict)
    saved_templates: List[str] = Field(description="Saved templates", default_factory=list)


class InteractionHistory(BaseModel):
    """Interaction history entry format"""
    timestamp: str = Field(description="Interaction timestamp")
    user_input: str = Field(description="User input")
    system_response: str = Field(description="System response")
    interaction_type: str = Field(description="Type of interaction")
    metadata: Dict[str, Any] = Field(description="Additional metadata", default_factory=dict)


class UserInteractionAgent(BaseAgent):
    """User Interaction Agent
    
    Features:
    - Natural language query processing
    - Multi-format response generation
    - Conversation history management
    - User preference tracking
    - Guided workflow assistance
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        super().__init__(
            name="User Interaction Agent",
            description="Manages user interactions and conversation flow",
            model_name=model_name,
            temperature=temperature
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._init_prompts()
        self.interaction_history = []
        self.user_preferences = UserPreferences()
        logger.info(f"{self.name} initialized")
        
    def _init_prompts(self) -> None:
        """Initialize prompts and chains"""
        self.query_analysis_parser = PydanticOutputParser(pydantic_object=QueryAnalysis)
        
        # Query analysis chain
        self.query_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Analyze the following user query:
                
                Query: {query}
                
                Please provide:
                1. The user's primary intent
                2. Key entities mentioned in the query
                3. Actions required to fulfill this query
                4. Your confidence in this analysis (0.0-1.0)
                5. A suggested response template
                
                {format_instructions}
                """,
                input_variables=["query"],
                partial_variables={"format_instructions": self.query_analysis_parser.get_format_instructions()}
            ),
            verbose=self.verbose
        )
        
        # Response generation chain
        self.response_generation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""Generate a helpful response to the user query:
                
                Query: {query}
                
                Analysis: {analysis}
                
                Current workflow state: {workflow_state}
                
                Please generate a clear, helpful response that addresses the user's needs.
                The response should be conversational but professional in tone.
                Include specific next steps or options when appropriate.
                """,
                input_variables=["query", "analysis", "workflow_state"]
            ),
            verbose=self.verbose
        )
        
        logger.debug("User interaction agent prompts and chains initialized")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def process_user_query(self, query_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user query and generate an appropriate response
        
        Args:
            query_text (str): The user's query text
            context (Dict[str, Any], optional): Additional context information
            
        Returns:
            Dict[str, Any]: Processing result including response
        """
        logger.info(f"Processing user query: {query_text[:50]}...")
        try:
            # Create query object
            query = UserQuery(
                query_text=query_text,
                timestamp=self._get_timestamp(),
                query_type=self._detect_query_type(query_text),
                context=context or {}
            )
            
            # Analyze query
            analysis_result = self.query_analysis_chain.invoke({"query": query_text})
            analysis = self.query_analysis_parser.parse(analysis_result["text"])
            
            # Generate response based on workflow state
            workflow_state = self.get_state().get("workflow_state", {})
            response = self.response_generation_chain.invoke({
                "query": query_text,
                "analysis": json.dumps(analysis.dict()),
                "workflow_state": json.dumps(workflow_state)
            })
            
            # Record interaction
            interaction = InteractionHistory(
                timestamp=self._get_timestamp(),
                user_input=query_text,
                system_response=response["text"],
                interaction_type=analysis.intent,
                metadata={"analysis": analysis.dict()}
            )
            self.interaction_history.append(interaction)
            
            # Update state
            self.update_state({
                "last_query": query.dict(),
                "last_response": response["text"],
                "interaction_count": len(self.interaction_history)
            })
            
            logger.info(f"Query processed successfully: {analysis.intent}")
            return {
                "success": True,
                "query": query.dict(),
                "analysis": analysis.dict(),
                "response": response["text"]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query_text": query_text
            }
    
    def _detect_query_type(self, query_text: str) -> str:
        """
        Detect the type of user query
        
        Args:
            query_text (str): The user's query text
            
        Returns:
            str: Query type (question, command, feedback)
        """
        query_text = query_text.lower()
        
        if query_text.endswith("?"):
            return "question"
        elif "how to" in query_text or "what is" in query_text:
            return "question"
        elif "please" in query_text or query_text.startswith(("create", "generate", "write", "make")):
            return "command"
        elif any(word in query_text for word in ["thanks", "good", "great", "bad", "terrible", "improve"]):
            return "feedback"
        else:
            return "general"
    
    def update_user_preferences(self, preferences: Dict[str, Any]) -> UserPreferences:
        """
        Update user preferences
        
        Args:
            preferences (Dict[str, Any]): New preference values
            
        Returns:
            UserPreferences: Updated preferences
        """
        logger.info("Updating user preferences")
        try:
            # Update only provided preferences
            current_prefs = self.user_preferences.dict()
            for key, value in preferences.items():
                if key in current_prefs:
                    current_prefs[key] = value
            
            # Create new preferences object
            self.user_preferences = UserPreferences(**current_prefs)
            
            # Update state
            self.update_state({"user_preferences": self.user_preferences.dict()})
            
            logger.info("User preferences updated successfully")
            return self.user_preferences
            
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
            return self.user_preferences
    
    def get_interaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent interaction history
        
        Args:
            limit (int, optional): Maximum number of entries to return
            
        Returns:
            List[Dict[str, Any]]: Recent interaction history
        """
        return [interaction.dict() for interaction in self.interaction_history[-limit:]]
    
    def generate_workflow_guidance(self, current_state: PaperWorkflowState) -> str:
        """
        Generate guidance based on current workflow state
        
        Args:
            current_state (PaperWorkflowState): Current workflow state
            
        Returns:
            str: Guidance message
        """
        logger.info("Generating workflow guidance")
        try:
            # Determine current stage
            stage = current_state.current_stage
            
            # Generate appropriate guidance
            if stage == "research":
                return "You're in the research phase. Consider refining your search queries or reviewing collected materials."
            elif stage == "writing":
                return "You're in the writing phase. Focus on developing your key arguments and maintaining a clear structure."
            elif stage == "editing":
                return "You're in the editing phase. Review for clarity, coherence, and adherence to your chosen style guide."
            elif stage == "review":
                return "You're in the review phase. Consider the feedback provided and make necessary revisions."
            else:
                return "Continue working through the current phase of your paper development."
                
        except Exception as e:
            logger.error(f"Error generating guidance: {str(e)}")
            return "Continue working on your paper. If you need help, please ask specific questions."
    
    def format_system_message(self, message: str, message_type: str = "info") -> str:
        """
        Format a system message for display
        
        Args:
            message (str): Message content
            message_type (str, optional): Message type (info, warning, error, success)
            
        Returns:
            str: Formatted message
        """
        if message_type == "info":
            prefix = "ℹ️ "
        elif message_type == "warning":
            prefix = "⚠️ "
        elif message_type == "error":
            prefix = "❌ "
        elif message_type == "success":
            prefix = "✅ "
        else:
            prefix = ""
            
        return f"{prefix}{message}"
    
    def run(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Run the user interaction agent
        
        Args:
            input_data (Dict[str, Any]): Input data
            config (Optional[RunnableConfig], optional): Run configuration
            
        Returns:
            Dict[str, Any]: Execution result
        """
        logger.info("User interaction agent running...")
        try:
            action = input_data.get("action", "")
            
            if action == "process_query":
                query_text = input_data.get("query_text", "")
                context = input_data.get("context", {})
                if not query_text:
                    raise ValueError("Query text not provided")
                return self.process_user_query(query_text, context)
                
            elif action == "update_preferences":
                preferences = input_data.get("preferences", {})
                if not preferences:
                    raise ValueError("Preferences not provided")
                updated_prefs = self.update_user_preferences(preferences)
                return {
                    "success": True,
                    "action": "update_preferences",
                    "preferences": updated_prefs.dict()
                }
                
            elif action == "get_history":
                limit = input_data.get("limit", 10)
                history = self.get_interaction_history(limit)
                return {
                    "success": True,
                    "action": "get_history",
                    "history": history
                }
                
            elif action == "get_guidance":
                workflow_state = input_data.get("workflow_state")
                if not workflow_state:
                    raise ValueError("Workflow state not provided")
                guidance = self.generate_workflow_guidance(workflow_state)
                return {
                    "success": True,
                    "action": "get_guidance",
                    "guidance": guidance
                }
                
            else:
                raise ValueError(f"Unsupported action: {action}")
                
        except Exception as e:
            logger.error(f"User interaction agent error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": input_data.get("action", "unknown")
            } 
