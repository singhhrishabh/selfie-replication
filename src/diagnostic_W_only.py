import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random

class FullRankAffineAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, h):
        return self.W(h) + self.b

def run_diagnostic():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    with open(os.path.join(data_dir, "split.json"), "r") as f:
        split = json.load(f)
    
    test_indices = split["test"]
    
    with open(os.path.join(data_dir, "topics.json"), "r") as f:
        dataset = json.load(f)
        
    act_data = torch.load(os.path.join(data_dir, "activations.pt"))
    vectors = act_data["contrastive_vectors"]
    d_model = vectors.shape[1]
    
    adapter = FullRankAffineAdapter(d_model)
    adapter.load_state_dict(torch.load(os.path.join(data_dir, "adapter.pt")))
    adapter.to(device)
    adapter.eval()
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()
    
    print("\n--- Diagnostic: W(h) Only (No Bias) ---")
    
    placeholder_char = "X"
    x_token_id = tokenizer.encode(placeholder_char, add_special_tokens=False)
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids]).to(device)
    
    base_embeds = model.get_input_embeddings()(full_ids)
    placeholder_positions = [i for i, tid in enumerate(prompt_ids) if tid in x_token_id]
    
    for i in range(min(5, len(test_indices))):
        idx = test_indices[i]
        topic = dataset[idx]["topic"]
        vec = vectors[idx].to(device, dtype=torch.bfloat16 if device == "mps" else torch.float32)
        
        # W(h) only
        adapted_vec = adapter.W(vec)
        
        inputs_embeds = base_embeds.clone()
        for pos in placeholder_positions:
            inputs_embeds[0, pos] = adapted_vec
                
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
        gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        clean_gen = gen.split('"')[0].split('\n')[0].strip()
        
        print(f"{topic:30s} -> {clean_gen}")

if __name__ == "__main__":
    run_diagnostic()
