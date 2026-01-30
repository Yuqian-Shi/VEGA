"""Config for language models."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMConfig:
    """A config for a language model.

    Attributes:
        provider: The name of the API provider.
        model: The name of the model.
        model_cls: The Python class corresponding to the model, mostly for
             Hugging Face transformers.
        tokenizer_cls: The Python class corresponding to the tokenizer, mostly
            for Hugging Face transformers.
        mode: The mode of the API calls, e.g., "chat" or "generation".
        multimodal_inputs: Whether the model supports multimodal inputs (vision).
    """

    provider: str
    model: str
    model_cls: type | None = None
    tokenizer_cls: type | None = None
    mode: str | None = None
    multimodal_inputs: bool = False
    gen_config: dict[str, Any] = dataclasses.field(default_factory=dict)


# def construct_llm_config(args: argparse.Namespace) -> LMConfig:
#     multimodal_inputs = getattr(args, 'multimodal_inputs', False)
#     llm_config = LMConfig(
#         provider=args.provider, 
#         model=args.model, 
#         mode=args.mode,
#         multimodal_inputs=multimodal_inputs
#     )
#     if args.provider in ["openai", "google", "api", "finetune"]:
#         llm_config.gen_config["temperature"] = args.temperature
#         llm_config.gen_config["top_p"] = args.top_p
#         llm_config.gen_config["context_length"] = args.context_length
#         llm_config.gen_config["max_tokens"] = args.max_tokens
#         llm_config.gen_config["stop_token"] = args.stop_token
#         llm_config.gen_config["max_obs_length"] = args.max_obs_length
#         llm_config.gen_config["max_retry"] = args.max_retry
#         # Add base_url for OpenAI provider
#         if args.provider == "openai":
#             llm_config.gen_config["base_url"] = args.model_endpoint
#     elif args.provider == "huggingface":
#         llm_config.gen_config["temperature"] = args.temperature
#         llm_config.gen_config["top_p"] = args.top_p
#         llm_config.gen_config["max_new_tokens"] = args.max_tokens
#         llm_config.gen_config["stop_sequences"] = (
#             [args.stop_token] if args.stop_token else None
#         )
#         llm_config.gen_config["max_obs_length"] = args.max_obs_length
#         llm_config.gen_config["model_endpoint"] = args.model_endpoint
#         llm_config.gen_config["max_retry"] = args.max_retry
#     else:
#         raise NotImplementedError(f"provider {args.provider} not implemented")
#     return llm_config
def construct_llm_config(config: dict) -> LMConfig:
    """Construct LMConfig from a dict or argparse.Namespace.
    
    Args:
        config: Dictionary or argparse.Namespace with LLM configuration
        
    Returns:
        LMConfig object
    """
    
    gen_cfg = config['generation']
    # Convert Namespace to dict if needed
    multimodal_inputs = config.get('multimodal_inputs', False)
    llm_config = LMConfig(
        provider=config['provider'], 
        model=config['model'], 
        mode=config.get('mode'),
        multimodal_inputs=multimodal_inputs
    )
    
    provider = config['provider']
    if provider in ["openai", "google", "api", "finetune", "claude", "gemini", "qwen", "vllm"]:
        llm_config.gen_config["temperature"] = gen_cfg['temperature']
        llm_config.gen_config["top_p"] = gen_cfg['top_p']
        llm_config.gen_config["context_length"] = gen_cfg['context_length']
        llm_config.gen_config["max_tokens"] = gen_cfg['max_tokens']
        llm_config.gen_config["stop_token"] = gen_cfg['stop_token']
        llm_config.gen_config["max_obs_length"] = gen_cfg['max_obs_length']
        llm_config.gen_config["max_retry"] = gen_cfg.get('max_retry', gen_cfg.get('max_retries', 1))
        llm_config.gen_config["api_key"] = gen_cfg.get('api_key') or config.get('api_key')
        # Add base_url for OpenAI provider
        if provider in ["openai", "vllm", "api"]:
            llm_config.gen_config["base_url"] = gen_cfg.get('base_url') or config.get('base_url') or config.get('endpoint')
    elif provider == "huggingface":
        llm_config.gen_config["temperature"] = gen_cfg['temperature']
        llm_config.gen_config["top_p"] = gen_cfg['top_p']
        llm_config.gen_config["max_new_tokens"] = gen_cfg['max_tokens']
        llm_config.gen_config["stop_sequences"] = (
            [gen_cfg['stop_token']] if gen_cfg['stop_token'] else None
        )
        llm_config.gen_config["max_obs_length"] = gen_cfg['max_obs_length']
        llm_config.gen_config["model_endpoint"] = gen_cfg.get('model_endpoint') or config.get('endpoint')
        llm_config.gen_config["max_retry"] = gen_cfg['max_retry']
    else:
        raise NotImplementedError(f"provider {provider} not implemented")
    
    return llm_config
