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

class XAIClient:
    """X.AI API와 상호작용하기 위한 클라이언트"""
    
    BASE_URL = "https://api.x.ai/v1"
    _instance = None  # 싱글톤 인스턴스
    
    @classmethod
    def get_instance(cls):
        """싱글톤 패턴을 구현한 인스턴스 getter"""
        if cls._instance is None:
            api_key = os.environ.get('XAI_API_KEY')
            if not api_key:
                raise ValueError("XAI_API_KEY 환경 변수가 설정되지 않았습니다")
            cls._instance = cls(api_key)
        return cls._instance
    
    def __init__(self, api_key):
        """
        Initialize the XAI client
        
        Args:
            api_key (str): X.AI API key starting with 'xai-'
        """
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def chat_completion(self, messages, model="grok-2-latest", temperature=0, stream=False):
        """
        Call the chat completions API
        
        Args:
            messages (list): List of message objects with 'role' and 'content'
            model (str): Model to use for completion
            temperature (float): Sampling temperature
            stream (bool): Whether to stream the response
            
        Returns:
            dict: The API response
        """
        endpoint = f"{self.BASE_URL}/chat/completions"
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream
        }
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"XAI API 호출 오류: {str(e)}")
            raise
        
    def generate_text(self, prompt):
        """
        Generate text using the XAI API
        
        Args:
            prompt (str): The prompt to generate text from
            
        Returns:
            str: The generated text
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.chat_completion(messages)
            return response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류 발생: {str(e)}")
            return ""

class XAITextGenerator(Runnable):
    """
    A Runnable wrapper for XAIClient.generate_text
    """
    def __init__(self, xai_client: XAIClient):
        self.xai_client = xai_client
    
    def invoke(self, prompt: str, config: Optional[RunnableConfig] = None, **kwargs) -> str:
        return self.xai_client.generate_text(prompt) 