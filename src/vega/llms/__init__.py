"""This module is adapt from https://github.com/zeno-ml/zeno-build"""
generate_from_gemini_completion = None

from .providers.hf_utils import generate_from_huggingface_completion

# OpenAI imports are optional
generate_from_openai_chat_completion = None
generate_from_openai_completion = None

try:
    from .providers.openai_utils import (
        generate_from_openai_chat_completion,
        generate_from_openai_completion,
    )
except Exception as e:
    print(f'OpenAI not set up, skipping import of providers.openai_utils.generate_from_openai_*: {e}')

from .providers.api_utils import (
    generate_with_api,
)
from .utils import call_llm
from . import lm_config

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_from_huggingface_completion",
    "generate_from_gemini_completion",
    "generate_with_api",
    "call_llm",
    "lm_config",
]
