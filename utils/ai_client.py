"""
AI API Client

This module provides a unified client for interacting with various AI APIs including OpenAI, xAI, and Anthropic.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
import logging

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

class AIClient:
    """통합된 AI API 클라이언트"""
    
    # API 엔드포인트
    XAI_API_URL = "https://api.x.ai/v1/chat/completions"
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
    
    # OpenAI 모델을 기본값으로 설정
    DEFAULT_MODEL = "gpt-3.5-turbo"
    FALLBACK_MODELS = ["gpt-4-turbo", "claude-3-7-sonnet-20240229"]
    MODEL_API_KEYS = {
        "grok-2": "XAI_API_KEY",
        "gpt-4-turbo": "OPENAI_API_KEY",
        "gpt-3.5-turbo": "OPENAI_API_KEY",
        "claude-3-7-sonnet-20240229": "ANTHROPIC_API_KEY"
    }
    MODEL_ENDPOINTS = {
        "grok-2": XAI_API_URL,
        "gpt-4-turbo": OPENAI_API_URL,
        "gpt-3.5-turbo": OPENAI_API_URL,
        "claude-3-7-sonnet-20240229": ANTHROPIC_API_URL
    }
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            # OpenAI API 키 사용
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                cls._instance = cls(api_key)
                cls._instance.current_model = cls.DEFAULT_MODEL
                logger.info(f"Using model: {cls.DEFAULT_MODEL}")
                return cls._instance
            else:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        return cls._instance
    
    def __init__(self, api_key: str):
        """Initialize with API key"""
        self.api_key = api_key
        self.current_model = self.DEFAULT_MODEL
        logger.info(f"AI Client initialized with model: {self.current_model}")
    
    def _check_model_availability(self):
        """Check if current model is available"""
        # 모델 체크 비활성화
        return True
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """채팅 완성 API 호출"""
        model_to_use = model or self.current_model
        
        # API 엔드포인트 및 헤더 설정
        if model_to_use == "grok-2":
            endpoint = self.XAI_API_URL
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "x-api-version": "2024-03"
            }
            data = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
                "stream": stream,
                "max_tokens": 4000
            }
        else:
            # OpenAI나 다른 API의 경우
            if "claude" in model_to_use:
                endpoint = self.ANTHROPIC_API_URL
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                }
                data = self._convert_to_anthropic_format(messages, model_to_use, temperature)
            else:
                endpoint = self.OPENAI_API_URL
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                data = {
                    "model": model_to_use,
                    "messages": messages,
                    "temperature": temperature,
                    "stream": stream
                }
        
        # Enhanced logging for debugging
        logger.debug(f"Using model: {model_to_use}")
        logger.debug(f"Endpoint: {endpoint}")
        
        # Log headers (excluding sensitive information)
        debug_headers = headers.copy()
        if 'Authorization' in debug_headers:
            debug_headers['Authorization'] = 'Bearer [MASKED]'
        if 'x-api-key' in debug_headers:
            debug_headers['x-api-key'] = '[MASKED]'
        logger.debug(f"Headers: {debug_headers}")
        
        try:
            # Add connection error handling
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=data,
                    timeout=30,
                    verify=True
                )
            except requests.exceptions.SSLError as ssl_err:
                logger.error(f"SSL Error: {str(ssl_err)}")
                raise
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Connection Error: {str(conn_err)}")
                raise
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"Timeout Error: {str(timeout_err)}")
                raise
            
            # Enhanced response logging
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                
                if response.status_code == 401:
                    logger.error("Authentication failed. Please check your API key.")
                elif response.status_code == 403:
                    logger.error("Access forbidden. Please check your API permissions.")
                elif response.status_code == 404:
                    logger.error("API endpoint or model not found. Please check the URL and model name.")
                elif response.status_code >= 500:
                    logger.error("Server error. The API service might be experiencing issues.")
                
                if self._is_model_error(error_msg) and model_to_use == "grok-2":
                    logger.warning(f"xAI API error with {model_to_use}: {error_msg}")
                    return self._try_fallback_chat_completion(messages, temperature, stream)
                raise Exception(error_msg)
            
            response_json = response.json()
            logger.debug(f"Response JSON: {response_json}")
            return response_json
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in chat_completion: {error_msg}")
            if self._is_model_error(error_msg) and model_to_use == "grok-2":
                logger.warning(f"Error with {model_to_use}: {error_msg}")
                return self._try_fallback_chat_completion(messages, temperature, stream)
            raise
    
    def _convert_to_anthropic_format(self, messages, model, temperature):
        """Anthropic API 형식으로 변환"""
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": "assistant" if msg["role"] == "assistant" else "user",
                    "content": msg["content"]
                })
        
        return {
            "messages": anthropic_messages,
            "model": model,
            "temperature": temperature,
            "system": system_message,
            "max_tokens": 4000
        }
    
    def _is_model_error(self, error_message: str) -> bool:
        """모델 관련 오류인지 확인"""
        error_patterns = [
            "model", "not found", "does not exist", "access",
            "unavailable", "unauthorized", "invalid", "quota", "limit"
        ]
        return any(pattern in error_message.lower() for pattern in error_patterns)
    
    def _try_fallback_chat_completion(self, messages, temperature, stream):
        """폴백 모델로 채팅 완성 재시도"""
        for model in self.FALLBACK_MODELS:
            try:
                if self._update_api_key_for_model(model):
                    self.current_model = model  # Update current model before trying
                    return self.chat_completion(messages, model, temperature, stream)
            except Exception as e:
                logger.warning(f"Fallback attempt with {model} failed: {str(e)}")
                continue
        raise ValueError("모든 폴백 모델이 실패했습니다.")
    
    def _update_api_key_for_model(self, model):
        """모델에 맞는 API 키로 업데이트"""
        key_name = self.MODEL_API_KEYS.get(model)
        if not key_name:
            return False
        
        api_key = os.environ.get(key_name)
        if not api_key:
            return False
        
        self.api_key = api_key
        return True
    
    def generate_text(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """텍스트 생성"""
        if isinstance(prompt, str):
            messages = [
                {"role": "user", "content": prompt}
            ]
        else:
            messages = prompt
        
        try:
            response = self.chat_completion(messages)
            return self._extract_content(response)
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {str(e)}")
            return f"Error: {str(e)}"
    
    def _extract_content(self, response: Any) -> str:
        """응답에서 컨텐츠 추출"""
        if isinstance(response, dict):
            if "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"]
            if "content" in response:
                return response["content"]
        if isinstance(response, str):
            return response
        return str(response)


class AITextGenerator(Runnable):
    """AI 텍스트 생성 모델 - Langchain Runnable 인터페이스 구현"""
    
    def __init__(self, temperature: float = 0.7):
        """Initialize the model"""
        self.temperature = temperature
        self.client = AIClient.get_instance()
    
    def invoke(
        self, 
        input_data: Union[str, List[BaseMessage], List[Dict[str, str]]],
        config: Optional[RunnableConfig] = None
    ) -> str:
        """Implement the invoke method required by Runnable"""
        # Handle different input types
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        elif isinstance(input_data, list) and all(isinstance(msg, BaseMessage) for msg in input_data):
            # Convert BaseMessage objects to dict format
            messages = []
            for msg in input_data:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    messages.append({"role": "system", "content": msg.content})
                else:
                    # Skip other message types
                    continue
        elif isinstance(input_data, list) and all(isinstance(msg, dict) for msg in input_data):
            messages = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Extract temperature from config if provided
        if config and "temperature" in config:
            temperature = config["temperature"]
        else:
            temperature = self.temperature
            
        # Call the API and return the result
        try:
            return self.client.generate_text(messages)
        except Exception as e:
            logger.error(f"Error in AITextGenerator: {str(e)}")
            return f"Error generating text: {str(e)}" 