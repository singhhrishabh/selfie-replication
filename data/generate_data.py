import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

TOPICS = [
    # ML / Tech
    "machine learning", "artificial intelligence", "backpropagation", "neural networks", 
    "automatic differentiation", "deep learning", "gradient descent", "transformers", 
    "reinforcement learning", "computer vision", "natural language processing",
    "stochastic gradient descent", "convolutional neural networks", "attention mechanism",
    "large language models", "generative adversarial networks", "blockchain", "cryptocurrency",
    "cloud computing", "cybersecurity", "quantum computing", "virtual reality", 
    "augmented reality", "internet of things", "5G networks", "autonomous vehicles",
    "3D printing", "robotics", "data science", "big data", "open source software",
    # Physics / Math
    "quantum mechanics", "general relativity", "special relativity", "string theory",
    "thermodynamics", "electromagnetism", "calculus", "linear algebra", "geometry",
    "number theory", "topology", "statistics", "probability", "chaos theory",
    "dark matter", "black holes", "the big bang theory", "particle physics",
    "standard model", "fluid dynamics", "optics", "acoustics", "astrophysics",
    # Biology / Med
    "photosynthesis", "cellular respiration", "DNA replication", "evolution",
    "genetics", "neuroscience", "immunology", "virology", "microbiology",
    "anatomy", "physiology", "pharmacology", "epidemiology", "biochemistry",
    "molecular biology", "ecology", "zoology", "botany", "marine biology",
    "CRISPR", "stem cells", "vaccines", "antibiotics", "cancer", "diabetes",
    # History
    "the roman empire", "ancient egypt", "ancient greece", "the middle ages",
    "the renaissance", "the industrial revolution", "world war I", "world war II",
    "the cold war", "the american revolution", "the french revolution", 
    "the russian revolution", "the enlightenment", "the age of discovery",
    "the vikings", "the aztec empire", "the inca empire", "the maya civilization",
    "the ming dynasty", "the ottoman empire", "the british empire",
    # Geography
    "mount everest", "the sahara desert", "the amazon rainforest", "the nile river",
    "the grand canyon", "the great barrier reef", "the mariana trench", 
    "the pacific ocean", "the atlantic ocean", "the antarctic", "the arctic",
    "the himalayas", "the alps", "the andes", "the rocky mountains",
    # Arts / Culture
    "the mona lisa", "impressionism", "surrealism", "cubism", "abstract expressionism",
    "the renaissance art", "classical music", "jazz", "rock and roll", "hip hop",
    "the beatles", "william shakespeare", "poetry", "the novel", "cinema",
    "photography", "architecture", "sculpture", "painting", "theater", "dance",
    # Philosophy / Religion
    "stoicism", "existentialism", "nihilism", "utilitarianism", "ethics",
    "epistemology", "metaphysics", "logic", "buddhism", "hinduism", "christianity",
    "islam", "judaism", "taoism", "confucianism", "shinto", "sikhism",
    # Economics / Politics
    "capitalism", "socialism", "communism", "democracy", "republic", "monarchy",
    "anarchism", "fascism", "macroeconomics", "microeconomics", "supply and demand",
    "inflation", "deflation", "interest rates", "the stock market", "international trade",
    # Sports
    "football", "basketball", "baseball", "tennis", "golf", "cricket", "rugby",
    "athletics", "swimming", "gymnastics", "boxing", "martial arts", "wrestling",
    # Everyday / Misc
    "coffee", "tea", "chocolate", "bread", "cheese", "wine", "beer", "water",
    "sleep", "dreams", "memory", "emotions", "happiness", "sadness", "anger",
    "fear", "love", "friendship", "family", "education", "work", "money",
    "time", "space", "energy", "matter", "light", "sound", "heat", "cold",
    "weather", "climate", "seasons", "day", "night", "sun", "moon", "stars",
    "planets", "galaxies", "the universe",
    # Additional random
    "the speed of light", "the speed of sound", "gravity", "friction", "inertia",
    "momentum", "force", "mass", "weight", "volume", "density", "temperature",
    "pressure", "entropy", "enthalpy", "kinetic energy", "potential energy",
    "work (physics)", "power (physics)", "voltage", "current", "resistance",
    "capacitance", "inductance", "magnetic field", "electric field", "photon",
    "electron", "proton", "neutron", "quark", "lepton", "boson", "fermion",
    "atom", "molecule", "ion", "isotope", "element", "compound", "mixture",
    "solution", "acid", "base", "salt", "pH", "catalyst", "enzyme", "protein",
    "carbohydrate", "lipid", "nucleic acid", "vitamin", "mineral", "hormone",
    "neuron", "synapse", "neurotransmitter", "brain", "heart", "lungs", "liver",
    "kidney", "stomach", "intestine", "muscle", "bone", "skin", "blood",
    "immune system", "nervous system", "endocrine system", "cardiovascular system",
    "respiratory system", "digestive system", "excretory system", "reproductive system",
    "skeletal system", "muscular system", "integumentary system",
    # Added some specific phrases
    "propagating gradients back through a neural network",
    "training a large language model",
    "baking a chocolate cake",
    "driving a manual transmission car",
    "playing the piano",
    "writing a python script",
    "solving a rubik's cube"
]

def generate_labels(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"Loading {model_name}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if device == "mps" else torch.float32).to(device)

    dataset = []

    print(f"Generating labels for {len(TOPICS)} topics...")
    for topic in tqdm(TOPICS):
        prompt = f"Write a single, short sentence describing what '{topic}' is. Do not include any extra commentary."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=40, temperature=0.7, do_sample=True, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Clean up output a bit
        generated = generated.replace('"', '').strip()
        if not generated:
            generated = topic
            
        dataset.append({
            "topic": topic,
            "label": generated
        })

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "topics.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved dataset to {out_path}")

if __name__ == "__main__":
    generate_labels()
