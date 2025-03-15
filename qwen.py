#!/usr/bin/env python
# coding: utf-8

import torch
torch.cuda.empty_cache()

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import argparse
import os
import json
import warnings
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process audio with Qwen-Audio-Chat model')
parser.add_argument('--dataset', type=str, required=True, choices=['AQA', 'SER', 'VSC'], 
                    help='Dataset folder to process (AQA, SER, or VSC)')
parser.add_argument('--output_dir', type=str, default='res', 
                    help='Directory to save results (default: res)')
args = parser.parse_args()

# Note: The default behavior now has injection attack prevention off.
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True, torch_dtype= torch.bfloat16 if device == "cuda" else torch.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

model.eval()
warnings.filterwarnings("ignore")

@torch.no_grad()
def respond(audio_path, input_text):    
    query = tokenizer.from_list_format([
        {'audio': audio_path}, # Either a local path or an url
        {'text': input_text},
    ])
    with torch.no_grad():
        response, history = model.chat(tokenizer, query=query, history=None)
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
    datasets_dir = ''
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_qwen.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)