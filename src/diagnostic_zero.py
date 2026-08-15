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

def test_zero():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()
    
    d_model = model.config.hidden_size
    
    adapter = ScalarAffineAdapter(d_model)
    adapter.load_state_dict(torch.load(os.path.join(data_dir, "adapter.pt")))
    adapter.to(device)
    adapter.eval()
    
    zero_vec = torch.zeros(d_model, device=device, dtype=torch.bfloat16 if device == "mps" else torch.float32)
    adapted_zero = adapter(zero_vec)
    
    placeholder_char = "Ω"
    placeholder_id = tokenizer.encode(placeholder_char, add_special_tokens=False)[0]
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids]).to(device)
    
    inputs_embeds = model.get_input_embeddings()(full_ids)
    
    for i, token_id in enumerate(prompt_ids):
        if token_id == placeholder_id:
            inputs_embeds[0, i] = adapted_zero
            
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        
    gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    clean_gen = gen.split('"')[0].split('\n')[0].strip()
    
    print("\n--- Zero-Vector Output ---")
    print(f"Trained adapter bias only output: {clean_gen}")

if __name__ == "__main__":
    test_zero()
