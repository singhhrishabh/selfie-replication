import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def layer_sweep():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if device == "mps" else torch.float32
    ).to(device)
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers, d_model = {model.config.hidden_size}")
    
    # 10 diverse topics
    sample_topics = [
        "backpropagation",
        "photosynthesis",
        "the roman empire",
        "quantum mechanics",
        "jazz",
        "capitalism",
        "DNA replication",
        "mount everest",
        "coffee",
        "machine learning",
    ]
    
    # Layers to test: early, mid-early, mid, mid-late, late
    layers_to_test = [4, 8, 12, 16, 20, 24]
    layers_to_test = [l for l in layers_to_test if l <= num_layers]
    
    # Extract activations at all layers in one pass per topic
    print(f"\nExtracting activations at layers {layers_to_test}...")
    all_raw_vecs = {layer: [] for layer in layers_to_test}
    
    for topic in tqdm(sample_topics, desc="Extracting"):
        prompt = f"Tell me about {topic}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        for layer in layers_to_test:
            vec = outputs.hidden_states[layer][0, -1, :].clone().cpu()
            all_raw_vecs[layer].append((topic, vec))
    
    # Compute per-layer contrastive vectors
    layer_contrastive = {}
    for layer in layers_to_test:
        vecs = torch.stack([v for _, v in all_raw_vecs[layer]])
        mean_vec = vecs.mean(dim=0)
        contrastive = vecs - mean_vec
        topics = [t for t, _ in all_raw_vecs[layer]]
        layer_contrastive[layer] = list(zip(topics, contrastive))
    
    # Build the explanation-seeking prompt for Llama
    # Llama 3.x uses a different chat template
    placeholder_char = "X"  # Use a simple token for Llama
    
    messages = [{"role": "user", "content": f'What is the meaning of "{placeholder_char}"?'}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_text += f'The meaning of "{placeholder_char}" is "'
    
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # Find placeholder positions
    placeholder_ids = tokenizer.encode(f'"{placeholder_char}"', add_special_tokens=False)
    # Find the single-char token for X
    x_token_id = tokenizer.encode(placeholder_char, add_special_tokens=False)
    print(f"\nPlaceholder '{placeholder_char}' encodes to token IDs: {x_token_id}")
    
    # Find all positions of the X token in prompt_ids
    placeholder_positions = [i for i, tid in enumerate(prompt_ids) if tid in x_token_id]
    print(f"Placeholder positions in prompt: {placeholder_positions}")
    
    # Get baseline embedding norm
    the_ids = tokenizer.encode("the", add_special_tokens=False)
    emb_norm = model.get_input_embeddings()(torch.tensor([the_ids]).to(device))[0, 0].float().norm().item()
    print(f"Reference: token embedding norm for 'the' = {emb_norm:.4f}")
    
    for layer in layers_to_test:
        print(f"\n{'='*80}")
        print(f"LAYER {layer}")
        print(f"{'='*80}")
        
        for topic, vec in layer_contrastive[layer]:
            vec_dev = vec.to(device, dtype=torch.bfloat16 if device == "mps" else torch.float32)
            vec_norm = vec.float().norm().item()
            
            # Normalize and scale to embedding magnitude
            h_hat = vec_dev / vec_dev.float().norm()
            scaled_vec = h_hat * emb_norm
            
            full_ids = torch.tensor([prompt_ids]).to(device)
            inputs_embeds = model.get_input_embeddings()(full_ids).clone()
            
            for pos in placeholder_positions:
                inputs_embeds[0, pos] = scaled_vec
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
            
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            clean_gen = gen.split('"')[0].split('\n')[0].strip()
            
            # Also try raw injection  
            inputs_embeds2 = model.get_input_embeddings()(full_ids).clone()
            for pos in placeholder_positions:
                inputs_embeds2[0, pos] = vec_dev
            
            with torch.no_grad():
                outputs2 = model.generate(
                    inputs_embeds=inputs_embeds2,
                    max_new_tokens=30,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
            
            gen2 = tokenizer.decode(outputs2[0], skip_special_tokens=True).strip()
            clean_gen2 = gen2.split('"')[0].split('\n')[0].strip()
            
            print(f"  {topic:25s} |norm={vec_norm:6.2f}| scaled={clean_gen[:50]:50s} | raw={clean_gen2[:50]}")

if __name__ == "__main__":
    layer_sweep()
