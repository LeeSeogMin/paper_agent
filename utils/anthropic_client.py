"""
Anthropic API Client

This module provides a client for interacting with the Anthropic API.
It implements a LangChain-compatible interface for the Anthropic API.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
import logging

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from config.settings import ANTHROPIC_MODEL, ANTHROPIC_TEMPERATURE
from config.api_keys import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

class ChatAnthropic:
    """
    Anthropic Chat model implementation for LangChain compatibility.
    """
    def __init__(
        self, 
        model: str = ANTHROPIC_MODEL, 
        temperature: float = ANTHROPIC_TEMPERATURE, 
        api_key: str = ANTHROPIC_API_KEY,
        verbose: bool = False
    ):
        """
        Initialize the Anthropic Chat model.
        
        Args:
            model (str): The Anthropic model to use (default: claude-3-7-sonnet-20240229)
            temperature (float): The temperature to use for generation
            api_key (str): The Anthropic API key
            verbose (bool): Whether to enable verbose logging
        """
        self.model = model
        self.model_name = model  # Add model_name for compatibility with LangChain
        self.temperature = temperature
        self.api_key = api_key
        self.verbose = verbose
        
        # Anthropic API endpoint
        self.api_endpoint = "https://api.anthropic.com/v1/messages"
        
        if not self.api_key:
            logger.warning("No Anthropic API key provided. API calls will fail.")
        
        if self.verbose:
            logger.info(f"Initialized Anthropic Chat model: {model} (temp: {temperature})")
    
    def _convert_messages_to_anthropic_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Convert LangChain messages to Anthropic API format.
        
        Args:
            messages (List[BaseMessage]): List of LangChain messages
            
        Returns:
            List[Dict[str, str]]: Messages in Anthropic API format
        """
        anthropic_messages = []
        system_message = None
        
        for message in messages:
            if isinstance(message, HumanMessage):
                anthropic_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif isinstance(message, AIMessage):
                anthropic_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
            elif isinstance(message, SystemMessage):
                # Anthropic handles system messages differently
                system_message = message.content
            else:
                logger.warning(f"Unsupported message type: {type(message)}. Skipping.")
        
        return anthropic_messages, system_message
    
    def invoke(self, messages: List[BaseMessage], **kwargs) -> Dict[str, Any]:
        """
        Invoke the Anthropic model with the given messages.
        
        Args:
            messages (List[BaseMessage]): List of LangChain messages
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dict[str, Any]: The model's response
        """
        if self.verbose:
            logger.info(f"Calling Anthropic model {self.model} with {len(messages)} messages")
        
        # Convert messages to Anthropic format
        anthropic_messages, system_message = self._convert_messages_to_anthropic_format(messages)
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            **kwargs
        }
        
        # Add system message if present
        if system_message:
            payload["system"] = system_message
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        try:
            # Make the API call
            response = requests.post(self.api_endpoint, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            # Extract the content from the response
            content = result["content"][0]["text"]
            
            return {"content": content}
            
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return {"content": f"Error: {str(e)}"}
    
    def generate(self, messages: List[BaseMessage], **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the Anthropic model.
        This is an alias for invoke() to maintain compatibility with different LangChain versions.
        
        Args:
            messages (List[BaseMessage]): List of LangChain messages
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dict[str, Any]: The model's response
        """
        return self.invoke(messages, **kwargs) 