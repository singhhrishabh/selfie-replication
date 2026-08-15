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
        # Initialize near identity
        self.W.weight.data = torch.eye(d_model) + torch.randn_like(self.W.weight.data) * 1e-4
        self.b = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, h):
        return self.W(h) + self.b

def train_adapter(model_name="meta-llama/Llama-3.2-3B-Instruct", epochs=5, lr=1e-3):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "topics.json"), "r") as f:
        dataset = json.load(f)
        
    act_data = torch.load(os.path.join(data_dir, "activations.pt"))
    vectors = act_data["contrastive_vectors"].to(device, dtype=torch.bfloat16 if device == "mps" else torch.float32)
    d_model = vectors.shape[1]
    
    # Shuffle and split
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Save split indices for eval
    with open(os.path.join(data_dir, "split.json"), "w") as f:
        json.dump({"train": train_indices, "test": test_indices}, f)
        
    # Setup adapter and optimizer
    adapter = FullRankAffineAdapter(d_model).to(device, dtype=torch.bfloat16 if device == "mps" else torch.float32)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=1e-2)
    
    # Prepare prompt template
    placeholder_char = "X"
    x_token_id = tokenizer.encode(placeholder_char, add_special_tokens=False)
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    placeholder_positions = [i for i, tid in enumerate(prompt_ids) if tid in x_token_id]
    
    print(f"Training adapter on {len(train_indices)} examples...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(train_indices)
        
        for idx in tqdm(train_indices, desc=f"Epoch {epoch+1}/{epochs}"):
            vec = vectors[idx]
            labels = dataset[idx]["labels"]
            
            # Adapted vector
            adapted_vec = adapter(vec)
            
            # Calculate loss for each label, average them
            idx_loss = 0.0
            for label in labels:
                label_ids = tokenizer.encode(label + '"', add_special_tokens=False)
                full_ids = torch.tensor([prompt_ids + label_ids]).to(device)
                
                # Get base embeddings
                inputs_embeds = model.get_input_embeddings()(full_ids).clone()
                
                # Inject adapted vector
                for pos in placeholder_positions:
                    inputs_embeds[0, pos] = adapted_vec
                        
                # Forward pass
                outputs = model(inputs_embeds=inputs_embeds)
                logits = outputs.logits
                
                shift_logits = logits[0, len(prompt_ids)-1 : -1, :].contiguous()
                shift_labels = torch.tensor(label_ids).to(device)
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
                idx_loss += loss
            
            idx_loss = idx_loss / len(labels)
            
            optimizer.zero_grad()
            idx_loss.backward()
            optimizer.step()
            
            epoch_loss += idx_loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_indices):.4f}")
        print(f"W Norm: {torch.norm(adapter.W.weight).item():.4f}, Bias Norm: {torch.norm(adapter.b).item():.4f}")

    # Save adapter
    torch.save(adapter.state_dict(), os.path.join(data_dir, "adapter.pt"))
    print("Adapter saved.")

if __name__ == "__main__":
    train_adapter()
