import json
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class FullRankAffineAdapter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, h):
        return self.W(h) + self.b

def generate_description(model, tokenizer, vec, device):
    placeholder_char = "X"
    x_token_id = tokenizer.encode(placeholder_char, add_special_tokens=False)
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = torch.tensor([prompt_ids]).to(device)
    
    base_embeds = model.get_input_embeddings()(full_ids)
    
    vec = vec.to(device, dtype=base_embeds.dtype)
    inputs_embeds = base_embeds.clone()
    
    placeholder_positions = [i for i, tid in enumerate(prompt_ids) if tid in x_token_id]
    
    for pos in placeholder_positions:
        inputs_embeds[0, pos] = vec
            
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=30,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        
    gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    clean_gen = gen.split('"')[0].split('\n')[0].strip()
    return clean_gen

def evaluate():
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
    mean_vector = act_data["mean_vector"]
    layer_idx = act_data["layer_idx"]
    
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
    
    print("Generating descriptions for test set...")
    untrained_texts = []
    trained_texts = []
    true_labels = []
    
    for idx in tqdm(test_indices):
        vec = vectors[idx]
        true_labels.append(dataset[idx]["labels"][0])
        
        # Untrained
        untrained_desc = generate_description(model, tokenizer, vec, device)
        untrained_texts.append(untrained_desc)
        
        # Trained
        adapted_vec = adapter(vec.to(device, dtype=torch.bfloat16 if device=="mps" else torch.float32))
        trained_desc = generate_description(model, tokenizer, adapted_vec, device)
        trained_texts.append(trained_desc)
        
    print("\nLoading sentence transformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Computing embeddings...")
    true_embeds = embedder.encode(true_labels, convert_to_tensor=True)
    untrained_embeds = embedder.encode(untrained_texts, convert_to_tensor=True)
    trained_embeds = embedder.encode(trained_texts, convert_to_tensor=True)
    
    def calc_recall(gen_embeds, true_embeds):
        from sentence_transformers.util import cos_sim
        sims = cos_sim(gen_embeds, true_embeds)
        
        recall_1 = 0
        recall_10 = 0
        n = len(sims)
        
        for i in range(n):
            _, indices = torch.sort(sims[i], descending=True)
            rank = (indices == i).nonzero().item()
            if rank == 0:
                recall_1 += 1
            if rank < 10:
                recall_10 += 1
                
        return recall_1 / n, recall_10 / n

    u_r1, u_r10 = calc_recall(untrained_embeds, true_embeds)
    t_r1, t_r10 = calc_recall(trained_embeds, true_embeds)
    
    print(f"\nResults on {len(test_indices)} held-out topics:")
    print(f"Untrained SelfIE: Recall@1 = {u_r1*100:.1f}%, Recall@10 = {u_r10*100:.1f}%")
    print(f"Trained Adapter:  Recall@1 = {t_r1*100:.1f}%, Recall@10 = {t_r10*100:.1f}%")
    
    print("\n--- Qualitative Examples ---")
    for i in range(min(5, len(test_indices))):
        idx = test_indices[i]
        print(f"Topic: {dataset[idx]['topic']}")
        print(f"True Label: {true_labels[i]}")
        print(f"Untrained:  {untrained_texts[i]}")
        print(f"Trained:    {trained_texts[i]}")
        print()
    
    results = {
        "untrained": {"recall@1": u_r1, "recall@10": u_r10},
        "trained": {"recall@1": t_r1, "recall@10": t_r10},
    }
    with open(os.path.join(data_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    evaluate()
