"""
xAI API Client

This module provides a client for interacting with the xAI API.
It implements a LangChain-compatible interface for the xAI API.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Generic
import logging

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.outputs import ChatGeneration, ChatResult
from config.settings import XAI_MODEL, XAI_TEMPERATURE
from config.api_keys import XAI_API_KEY

logger = logging.getLogger(__name__)

# Define input and output types
Input = TypeVar('Input')
Output = TypeVar('Output')

class ChatXAI(Runnable[List[BaseMessage], Dict[str, Any]]):
    """
    xAI Chat model implementation for LangChain compatibility.
    """
    def __init__(
        self, 
        model: str = XAI_MODEL, 
        temperature: float = XAI_TEMPERATURE, 
        api_key: str = XAI_API_KEY,
        verbose: bool = False
    ):
        """
        Initialize the xAI Chat model.
        
        Args:
            model (str): The xAI model to use (default: grok-2)
            temperature (float): The temperature to use for generation
            api_key (str): The xAI API key
            verbose (bool): Whether to enable verbose logging
        """
        self.model = model
        self.model_name = model  # Add model_name for compatibility with LangChain
        self.temperature = temperature
        self.api_key = api_key
        self.verbose = verbose
        
        # 실제 Grok API 엔드포인트
        self.api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No xAI API key provided. API calls will fail.")
        
        if self.verbose:
            logger.info(f"Initialized xAI Chat model: {model} (temp: {temperature})")
    
    def _convert_messages_to_xai_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to xAI API format.
        
        Args:
            messages (List[BaseMessage]): List of LangChain messages
            
        Returns:
            List[Dict[str, str]]: Messages in xAI API format
        """
        xai_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                logger.warning(f"Unsupported message type: {type(message)}. Skipping.")
                continue
            
            xai_messages.append({
                "role": role,
                "content": message.content
            })
        
        return xai_messages
    
    def invoke(self, messages: List[BaseMessage], config: Optional[RunnableConfig] = None, **kwargs) -> Dict[str, Any]:
        """
        Invoke the xAI model with the given messages.
        
        Args:
            messages (List[BaseMessage]): List of LangChain messages
            config (Optional[RunnableConfig]): Configuration for the runnable
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dict[str, Any]: The model's response
        """
        if self.verbose:
            logger.info(f"Calling xAI model {self.model} with {len(messages)} messages")
        
        # Convert messages to xAI format
        xai_messages = self._convert_messages_to_xai_format(messages)
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": xai_messages,
            "temperature": self.temperature,
            **kwargs
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # 실제 API 호출
            logger.info(f"Making request to Grok API with model: {self.model}")
            response = requests.post(self.api_endpoint, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from the response
            content = result["choices"][0]["message"]["content"]
            
            # Create an AIMessage for compatibility with LangChain
            ai_message = AIMessage(content=content)
            
            return {"output": ai_message}
            
        except Exception as e:
            logger.error(f"Error calling xAI API: {str(e)}")
            return {"output": AIMessage(content=f"Error: {str(e)}")}
    
    def generate(self, messages: List[BaseMessage], **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the xAI model.
        This is an alias for invoke() to maintain compatibility with different LangChain versions.
        
        Args:
            messages (List[BaseMessage]): List of LangChain messages
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dict[str, Any]: The model's response
        """
        return self.invoke(messages, **kwargs) 