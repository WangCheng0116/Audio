#!/usr/bin/env python
# coding: utf-8

import torch
torch.cuda.empty_cache()

import torch
import torchaudio
import transformers
from gazelle import GazelleForConditionalGeneration
import argparse
import os
import json
import warnings
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process audio with Gazelle model')
parser.add_argument('--dataset', type=str, required=True, choices=['AQA', 'SER', 'VSC'], 
                    help='Dataset folder to process (AQA, SER, or VSC)')
parser.add_argument('--output_dir', type=str, default='res', 
                    help='Directory to save results (default: res)')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model_id = "tincans-ai/gazelle-v0.2"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = GazelleForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype
).to(device)

audio_processor = transformers.Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

warnings.filterwarnings("ignore")

@torch.no_grad()
def respond(audio_path, input_text):    
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    
    audio_values = audio_processor(
        audio=audio, return_tensors="pt", sampling_rate=16000
    ).input_values.squeeze(0).to(model.device).to(model.dtype)
    
    if "<|audio|>" not in input_text:
        input_text = input_text + " \n<|audio|>"
    
    msgs = [{"role": "user", "content": input_text}]
    input_ids = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    outputs = model.generate(
        audio_values=audio_values,
        input_ids=input_ids,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id
    )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response_text = full_text.split("[/INST]")[-1].replace("</s>", "").strip()
    
    return response_text

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
    datasets_dir = 'datasets'
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_gazelle.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)