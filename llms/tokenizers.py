from typing import Any

import tiktoken
from transformers import LlamaTokenizer, AutoTokenizer  # type: ignore
from .providers.vllm_utils import load_vllm_tokenizer


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except: # The provider is in openai format but the model is a finetuned model
                self.tokenizer = None
        elif provider == "huggingface" or provider == "vllm":
            self.tokenizer = load_vllm_tokenizer(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider in ["google", "api", "finetune"]:
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
