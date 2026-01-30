import argparse
import re
from typing import Any

try:
    from vertexai.preview.generative_models import Image
    from vega.llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')
    Image = None

from vega.llms import (
    generate_from_huggingface_completion,
    generate_with_api,
    lm_config,
)

try:
    from vega.llms import (
        generate_from_openai_chat_completion,
        generate_from_openai_completion,
    )
except ImportError:
    print('OpenAI not set up, generate_from_openai functions unavailable')
    generate_from_openai_chat_completion = None
    generate_from_openai_completion = None

APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    api_key = None,
    base_url =  None,
) -> str:
    response: str
    if lm_config.provider in ["openai", "vllm", "api"]:
        # Use base_url from config if not explicitly provided
        if base_url is None:
            base_url = lm_config.gen_config.get("base_url")
        if api_key is None:
            api_key = lm_config.gen_config.get("api_key")

        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                base_url=base_url,
                api_key=api_key,
            )
            # Remove thinking process if present
            if isinstance(response, str):
                cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
                if not cleaned_response and response:
                    print(f"WARNING: Response became empty after stripping <think> tags. Original response length: {len(response)}")
                    # If stripping results in empty response, keep the original
                    pass
                else:
                    response = cleaned_response
        elif lm_config.mode == "completion":
            if generate_from_openai_completion is None:
                raise RuntimeError("OpenAI not set up. Install openai package: pip install openai")
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
                api_key=api_key,
                base_url=base_url
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
        )
    elif lm_config.provider in ["finetune", "claude", "gemini", "qwen"]:
        args = {
            "temperature": lm_config.gen_config["temperature"],   # openai, gemini, claude
            "max_tokens": lm_config.gen_config["max_tokens"],     # openai, gemini, claude
            "top_k": lm_config.gen_config["top_p"],               # qwen
            "api_key": lm_config.gen_config.get("api_key"),
            "base_url": lm_config.gen_config.get("base_url"),
            "provider": lm_config.provider
        }
        response = generate_with_api(prompt, lm_config.model, args)

    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
