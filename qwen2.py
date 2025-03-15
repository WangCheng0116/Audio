import torch
torch.cuda.empty_cache()

import librosa
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import argparse
import os
import json
import warnings
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process audio with Qwen2 model')
parser.add_argument('--dataset', type=str, required=True, choices=['AQA', 'SER', 'VSC'], 
                    help='Dataset folder to process (AQA, SER, or VSC)')
parser.add_argument('--output_dir', type=str, default='res', 
                    help='Directory to save results (default: res)')
args = parser.parse_args()

# Load the model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.eval()
warnings.filterwarnings("ignore")

@torch.no_grad()
def respond(audio_path, input_text):
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
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=100)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

def process_aqa_data(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        aqa_data = json.load(f)
    
    base_dir = os.path.dirname(input_json_path)
    results = []
    
    for item in tqdm(aqa_data, desc="Processing audio files"):
        audio_path = os.path.join(base_dir, item['audio'])
        
        result = {
            'audio': item['audio'],
            'gt': item['gt']
        }
        
        prompt_types = ['faithful', 'adversarial', 'irrelevant', 'neutral']
        for prompt_type in prompt_types:
            if prompt_type in item:
                response = respond(audio_path, item[prompt_type])
                result[f'{prompt_type}_response'] = response
        results.append(result)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete. Results saved to {output_json_path}")

if __name__ == "__main__":
    # Determine the input JSON file based on the dataset argument
    dataset_folder = args.dataset
    json_file = args.dataset + '.json'
    
    # Construct paths
    datasets_dir = os.path.dirname(os.path.abspath(__file__)) 
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_qwen2.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)