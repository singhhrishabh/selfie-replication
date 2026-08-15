import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ScalarAffineAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, h):
        return self.alpha * h + self.b

def test_alpha_only():
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
    d_model = vectors.shape[1]
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()
    
    adapter = ScalarAffineAdapter(d_model)
    adapter.load_state_dict(torch.load(os.path.join(data_dir, "adapter.pt")))
    adapter.to(device)
    adapter.eval()
    
    print(f"\nTrained alpha: {adapter.alpha.item():.4f}")
    print(f"Trained bias norm: {torch.norm(adapter.b.float()).item():.4f}")
    
    placeholder_char = "Ω"
    placeholder_id = tokenizer.encode(placeholder_char, add_special_tokens=False)[0]
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    print("\n--- Alpha-Only Outputs (no bias) ---")
    for i in range(5):
        idx = test_indices[i]
        topic = dataset[idx]["topic"]
        vec = vectors[idx].to(device, dtype=torch.bfloat16 if device == "mps" else torch.float32)
        
        # Alpha * h only, no bias
        adapted_vec = adapter.alpha * vec
        
        full_ids = torch.tensor([prompt_ids]).to(device)
        inputs_embeds = model.get_input_embeddings()(full_ids)
        
        for j, tid in enumerate(prompt_ids):
            if tid == placeholder_id:
                inputs_embeds[0, j] = adapted_vec
                
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=30,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            
        gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        clean_gen = gen.split('"')[0].split('\n')[0].strip()
        
        print(f"\nTopic: {topic}")
        print(f"  alpha*h norm: {torch.norm(adapted_vec.float()).item():.4f}")
        print(f"  Output: {clean_gen}")

if __name__ == "__main__":
    test_alpha_only()
