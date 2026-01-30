"""Tools to generate from OpenAI prompts."""
import functools
import logging
import os
import random
import time
from typing import Any, List, Optional, Union, Dict

import openai
from openai import OpenAI
import tiktoken

RetryError = Exception

@functools.lru_cache(maxsize=None)
def _get_openai_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)

def generate_from_openai_chat_completion(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    context_length: int = 16384,
    stop_token: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> str:
    if model_config is None:
        model_config = {}

    if api_key:
        api_key_val = api_key
    elif "api_key" in model_config:
        api_key_val = model_config["api_key"]
    elif "generation" in model_config and "api_key" in model_config["generation"]:
        api_key_val = model_config["generation"]["api_key"]
    else:
        api_key_val = "EMPTY"

    if not api_key_val:
        raise ValueError("API Key not found in config arguments (environment variables ignored)")
    
    if base_url:
         base_url_val = base_url
    elif "base_url" in model_config:
         base_url_val = model_config["base_url"]
    elif "generation" in model_config and "base_url" in model_config["generation"]:
         base_url_val = model_config["generation"]["base_url"]
    else:
         base_url_val = None

    client = _get_openai_client(api_key_val, base_url_val)
    


    if stop_token:
        # OpenAI Python SDK allows list or str, usually
        stop = [stop_token]
    else:
        stop = None
        
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        logging.error(f"OpenAI Generation Error: {e}")
        raise e


def generate_from_openai_completion(
    prompt: Union[str, List[Any]],
    model_config: Dict[str, Any],
    temperature: float = 1.0,
    max_tokens: int = 512,
    top_p: float = 1.0,
    stop_token: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    if model_config is None:
        model_config = {}

    if "api_key" in model_config:
        api_key_val = model_config["api_key"]
    elif "generation" in model_config and "api_key" in model_config["generation"]:
        api_key_val = model_config["generation"]["api_key"]
    elif api_key:
        api_key_val = api_key
    else:
        api_key_val = os.environ.get("OPENAI_API_KEY")

    if not api_key_val:
        raise ValueError("OPENAI_API_KEY not found in config or environment")

    # Check for generation sub-dict if keys are missing
    if "base_url" in model_config:
         base_url_val = model_config["base_url"]
    elif "generation" in model_config and "base_url" in model_config["generation"]:
         base_url_val = model_config["generation"]["base_url"]
    else:
         base_url_val = base_url if base_url else os.environ.get("OPENAI_API_URL")

    client = _get_openai_client(api_key_val, base_url_val)
    
    if "model" in model_config:
        model = model_config["model"]
    elif "generation" in model_config and "model" in model_config["generation"]:
        model = model_config["generation"]["model"]
    else:
        # Default fallback
        model = "gpt-3.5-turbo-instruct"

    if isinstance(prompt, list):
        # some old legacy code might pass list of strings
        # but modern chat models usually prefer single prompts or message lists.
        # Here we assume text completion endpoint usage.
        prompt = prompt[0] if prompt else ""

    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_token,
        )
        answer = response.choices[0].text
        return answer
    except Exception as e:
        logging.error(f"OpenAI Generation Error: {e}")
        raise e

