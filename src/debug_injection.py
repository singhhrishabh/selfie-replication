import torch
from transformers import AutoTokenizer

def debug_injection():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    placeholder_char = "Ω"
    placeholder_id = tokenizer.encode(placeholder_char, add_special_tokens=False)[0]
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    
    print("\n--- STEP 1: Naive Token Matching ---")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"Full prompt text:\n{repr(prompt_text)}")
    print(f"\nPrompt IDs:\n{prompt_ids}")
    
    matches = [i for i, tid in enumerate(prompt_ids) if tid == placeholder_id]
    print(f"\nPlaceholder ID ({placeholder_id}) appears {len(matches)} time(s) in prompt_ids.")
    for i in matches:
        decoded_match = tokenizer.decode([prompt_ids[i]])
        print(f"  Match at index {i} decodes to: {repr(decoded_match)}")
        
    print("\n--- STEP 2: Offset-Based Matching ---")
    # We need to tokenize with offset mapping. Fast tokenizers support this.
    encoding = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    
    # Find character indices of all occurrences of placeholder_char in prompt_text
    char_indices = []
    idx = prompt_text.find(placeholder_char)
    while idx != -1:
        char_indices.append(idx)
        idx = prompt_text.find(placeholder_char, idx + 1)
        
    print(f"Found placeholder '{placeholder_char}' at character indices: {char_indices}")
    
    # Find which tokens overlap with these character indices
    token_indices = []
    for i, (start, end) in enumerate(offsets):
        for ci in char_indices:
            # If the character falls within the token's span
            if start <= ci < end:
                if i not in token_indices:
                    token_indices.append(i)
                    
    print(f"\nOffset-based matching found tokens at indices: {token_indices}")
    for i in token_indices:
        decoded_tok = tokenizer.decode([prompt_ids[i]])
        tok_id = prompt_ids[i]
        print(f"  Token at index {i} (ID: {tok_id}) decodes to: {repr(decoded_tok)}")

if __name__ == "__main__":
    debug_injection()
