import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def layer_sweep():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "topics.json"), "r") as f:
        dataset = json.load(f)
    
    # Pick 10 diverse topics by hand (spread across categories)
    sample_topics = [
        "backpropagation",
        "photosynthesis",
        "the roman empire",
        "quantum mechanics",
        "jazz",
        "capitalism",
        "DNA replication",
        "mount everest",
        "chess",
        "coffee",
    ]
    # Find indices in dataset; fall back to first 10 if not found
    sample_indices = []
    for t in sample_topics:
        idx = next((i for i, item in enumerate(dataset) if item["topic"] == t), None)
        if idx is not None:
            sample_indices.append(idx)
    if len(sample_indices) < 10:
        # Fill with whatever we have
        for i in range(len(dataset)):
            if i not in sample_indices:
                sample_indices.append(i)
            if len(sample_indices) >= 10:
                break
    
    sample_indices = sample_indices[:10]
    print(f"Using {len(sample_indices)} topics for sweep")
    
    layers_to_test = [8, 12, 16, 20, 24]
    
    # --- Step 1: Extract activations at each layer for the 10 topics ---
    # We'll extract all layers in one pass per topic to be efficient
    print("\nExtracting activations...")
    # layer -> list of (topic_name, contrastive_vec)
    all_raw_vecs = {layer: [] for layer in layers_to_test}
    
    for idx in tqdm(sample_indices, desc="Extracting"):
        topic = dataset[idx]["topic"]
        prompt = f"Tell me about {topic}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        for layer in layers_to_test:
            vec = outputs.hidden_states[layer][0, -1, :].clone().cpu()
            all_raw_vecs[layer].append((topic, vec))
    
    # Compute per-layer means and subtract
    layer_contrastive = {}
    for layer in layers_to_test:
        vecs = torch.stack([v for _, v in all_raw_vecs[layer]])
        mean_vec = vecs.mean(dim=0)
        contrastive = vecs - mean_vec
        topics = [t for t, _ in all_raw_vecs[layer]]
        layer_contrastive[layer] = list(zip(topics, contrastive))
    
    # --- Step 2: For each layer, inject each contrastive vector and generate ---
    placeholder_char = "Ω"
    placeholder_id = tokenizer.encode(placeholder_char, add_special_tokens=False)[0]
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # Get baseline embedding norm for reference
    the_id = tokenizer.encode("the", add_special_tokens=False)[0]
    emb_norm = model.get_input_embeddings()(torch.tensor([[the_id]]).to(device))[0, 0].float().norm().item()
    
    print(f"\nReference: token embedding norm for 'the' = {emb_norm:.4f}")
    
    for layer in layers_to_test:
        print(f"\n{'='*70}")
        print(f"LAYER {layer}")
        print(f"{'='*70}")
        
        for topic, vec in layer_contrastive[layer]:
            vec_dev = vec.to(device, dtype=torch.bfloat16 if device == "mps" else torch.float32)
            vec_norm = vec.float().norm().item()
            
            # Normalize to embedding-scale magnitude
            h_hat = vec_dev / vec_dev.float().norm()
            # Try scale that matches typical embedding norm
            scaled_vec = h_hat * emb_norm
            
            full_ids = torch.tensor([prompt_ids]).to(device)
            inputs_embeds = model.get_input_embeddings()(full_ids).clone()
            
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
            clean_gen = gen.split('"')[0].split('\n')[0].strip()
            
            # Also try raw (unscaled) injection
            inputs_embeds2 = model.get_input_embeddings()(full_ids).clone()
            for i, tid in enumerate(prompt_ids):
                if tid == placeholder_id:
                    inputs_embeds2[0, i] = vec_dev
            
            with torch.no_grad():
                outputs2 = model.generate(
                    inputs_embeds=inputs_embeds2,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
            
            gen2 = tokenizer.decode(outputs2[0], skip_special_tokens=True).strip()
            clean_gen2 = gen2.split('"')[0].split('\n')[0].strip()
            
            print(f"  {topic:45s} |norm={vec_norm:6.2f}| scaled={clean_gen:40s} | raw={clean_gen2}")

if __name__ == "__main__":
    layer_sweep()
