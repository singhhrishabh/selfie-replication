import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_diagnostic():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    with open(os.path.join(data_dir, "topics.json"), "r") as f:
        dataset = json.load(f)
    with open(os.path.join(data_dir, "split.json"), "r") as f:
        split = json.load(f)
        
    test_indices = split["test"]
    
    act_data = torch.load(os.path.join(data_dir, "activations.pt"))
    vectors = act_data["contrastive_vectors"]
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()
    
    # 1. Print norms
    print("\n--- Diagnostic: Norms ---")
    raw_vec = vectors[test_indices[0]]
    print(f"L2 norm of one raw contrastive vector: {torch.norm(raw_vec.float()).item():.4f}")
    
    token_id = tokenizer.encode("the", add_special_tokens=False)[0]
    full_ids = torch.tensor([[token_id]]).to(device)
    token_embed = model.get_input_embeddings()(full_ids)[0, 0]
    print(f"L2 norm of model's token embedding for 'the': {torch.norm(token_embed.float()).item():.4f}")
    
    # 2. Multi-scale generation protocol
    def generate_description_multi_scale(vec):
        placeholder_char = "Ω"
        placeholder_id = tokenizer.encode(placeholder_char, add_special_tokens=False)[0]
        
        messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_text += f'The meaning of "{placeholder_char}" is "'
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = torch.tensor([prompt_ids]).to(device)
        base_embeds = model.get_input_embeddings()(full_ids)
        
        # Normalize to unit L2
        vec = vec.to(device, dtype=base_embeds.dtype)
        h_hat = vec / torch.norm(vec.float())
        
        scales = [1, 2, 4, 8, 16, 32]
        outputs_dict = {}
        
        for scale in scales:
            scaled_vec = h_hat * scale
            inputs_embeds = base_embeds.clone()
            
            for i, tid in enumerate(prompt_ids):
                if tid == placeholder_id:
                    inputs_embeds[0, i] = scaled_vec
                    
            with torch.no_grad():
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
                
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            # Extract just the first part before a closing quote or newline
            clean_gen = gen.split('"')[0].split('\n')[0].strip()
            outputs_dict[scale] = clean_gen
            
        return outputs_dict
        
    print("\n--- Diagnostic: Multi-scale Output ---")
    for i in range(5):
        idx = test_indices[i]
        topic = dataset[idx]["topic"]
        vec = vectors[idx]
        
        print(f"\nTopic: {topic}")
        results = generate_description_multi_scale(vec)
        for scale, text in results.items():
            print(f"  Scale {scale:2d}: {text}")

if __name__ == "__main__":
    run_diagnostic()
