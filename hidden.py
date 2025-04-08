import torch
import os

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()

import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import argparse
import json
import warnings
from tqdm import tqdm
import numpy as np
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process audio with Qwen2 model and extract hidden states')
parser.add_argument('--output_dir', type=str, default='hidden_states_output', 
                    help='Directory to save hidden states (default: hidden_states_output)')
parser.add_argument('--samples', type=int, default=1000,
                    help='Number of samples to process (default: 100)')
args = parser.parse_args()

# Load the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", 
    device_map="auto", 
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, 
    output_hidden_states=True  # Enable output of hidden states
)

model.eval()
warnings.filterwarnings("ignore")

@torch.no_grad()
def forward_with_hidden_states(audio_path, input_text):
    # Load audio from local file
    sampling_rate = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(audio_path, sr=sampling_rate)
    
    # Create conversation structure
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": input_text}
        ]}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True, sampling_rate=sampling_rate)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    # Get the hidden states
    hidden_states = outputs.hidden_states
    
    return hidden_states

def process_aqa_data(input_json_path, output_dir, max_samples=100):
    with open(input_json_path, 'r') as f:
        aqa_data = json.load(f)
    
    # Limit samples
    aqa_data = aqa_data[:max_samples]
    
    base_dir = os.path.dirname(input_json_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first sample to determine the number of layers
    first_item = aqa_data[0]
    first_audio_path = os.path.join(base_dir, first_item['audio'])
    
    # Just use any prompt type that exists
    for prompt_type in ['faithful', 'adversarial', 'irrelevant', 'neutral']:
        if prompt_type in first_item:
            first_hidden_states = forward_with_hidden_states(first_audio_path, first_item[prompt_type])
            break
    
    # Determine which layers to save
    total_layers = len(first_hidden_states)
    layer_indices = [1, 
                    #  total_layers // 2, 
                    #  total_layers - 1
                     ]  # First, middle, last
    print(f"Total layers: {total_layers}, Saving layers: {layer_indices}")
    
    # Initialize dictionaries to collect data for each type and layer
    collected_data = {}
    prompt_types = ['faithful', 'adversarial', 'irrelevant', 'neutral']
    
    for prompt_type in prompt_types:
        for layer_idx in layer_indices:
            key = f"{prompt_type}_{layer_idx}"
            collected_data[key] = []
    
    # Process each audio file
    for idx, item in enumerate(tqdm(aqa_data, desc="Processing audio files")):
        audio_path = os.path.join(base_dir, item['audio'])
        
        for prompt_type in prompt_types:
            if prompt_type in item:
                # Get hidden states
                hidden_states = forward_with_hidden_states(audio_path, item[prompt_type])
                
                # Save specified layers (only the last position)
                for layer_idx in layer_indices:
                    # Get hidden states for this layer, last position only
                    last_position_hidden_state = hidden_states[layer_idx][:, -1, :]
                    
                    # Convert BFloat16 to Float32 before converting to numpy
                    np_hidden_states = last_position_hidden_state.cpu().to(torch.float32).numpy()
                    
                    # Add to collected data
                    key = f"{prompt_type}_{layer_idx}"
                    collected_data[key].append(np_hidden_states)
        
        # Free up memory
        torch.cuda.empty_cache()
    
    # Save all collected data to separate NPY files
    for key, data_list in collected_data.items():
        if data_list:  # Only save if we have data
            # Stack all examples for this type and layer
            stacked_data = np.vstack(data_list)
            
            # Save to NPY file
            output_path = os.path.join(output_dir, f"{key}_qwen.npy")
            np.save(output_path, stacked_data)
            print(f"Saved {stacked_data.shape} to {output_path}")
    
    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":    
    # Fixed dataset to AQA
    dataset_folder = "AQA"
    json_file = dataset_folder + '.json'
    
    # Construct paths
    datasets_dir = os.path.dirname(os.path.abspath(__file__)) 
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)
    output_dir = args.output_dir
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Results will be saved to: {output_dir}")
    print(f"Processing first {args.samples} samples only")
    
    process_aqa_data(input_json_path, output_dir, max_samples=args.samples)
