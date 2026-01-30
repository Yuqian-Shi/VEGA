from transformers import AutoTokenizer, PreTrainedTokenizer

def load_vllm_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load a tokenizer compatible with VLLM, applying necessary patches for specific models.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Patch for ChatGLM4 tokenizer issue where _pad() does not accept padding_side
    # This is a known issue when using transformers >= 4.34 with ChatGLM4
    if "ChatGLM4Tokenizer" in type(tokenizer).__name__:
        original_pad = tokenizer._pad

        def _pad_wrapper(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask, padding_side=None):
            return original_pad(encoded_inputs, max_length, padding_strategy, pad_to_multiple_of, return_attention_mask)

        tokenizer._pad = _pad_wrapper
    
    return tokenizer
