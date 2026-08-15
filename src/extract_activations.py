import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def extract_activations(model_name="Qwen/Qwen2.5-1.5B-Instruct", layer_idx=14):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if device == "mps" else torch.float32).to(device)

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "topics.json")
    with open(data_path, "r") as f:
        dataset = json.load(f)

    vectors = []
    
    print(f"Extracting activations at layer {layer_idx}...")
    for item in tqdm(dataset):
        topic = item["topic"]
        prompt = f"Tell me about {topic}"
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        # Hidden states: tuple of length num_layers + 1
        # Shape of each hidden state: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.hidden_states[layer_idx]
        
        # Get the activation at the final token position (index -1)
        # We want the vector before generation starts, which is the last token of the prompt.
        final_token_activation = hidden_states[0, -1, :].clone().cpu()
        vectors.append(final_token_activation)
        
    # Stack all vectors
    vectors_tensor = torch.stack(vectors)
    
    # Compute the mean activation across all topics
    mean_vector = vectors_tensor.mean(dim=0)
    
    # Subtract the mean to get contrastive topic directions
    contrastive_vectors = vectors_tensor - mean_vector
    
    # Save the vectors
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    
    torch.save({
        "contrastive_vectors": contrastive_vectors,
        "mean_vector": mean_vector,
        "layer_idx": layer_idx
    }, os.path.join(out_dir, "activations.pt"))
    
    print(f"Saved {contrastive_vectors.shape[0]} vectors of dimension {contrastive_vectors.shape[1]}")

if __name__ == "__main__":
    extract_activations()
