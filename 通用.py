import torch
torch.cuda.empty_cache()


import argparse
import os
import json
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process audio with Gazelle model')
parser.add_argument('--dataset', type=str, required=True, choices=['AQA', 'SER', 'VSC'], 
                    help='Dataset folder to process (AQA, SER, or VSC)')
parser.add_argument('--output_dir', type=str, default='res', 
                    help='Directory to save results (default: res)')
args = parser.parse_args()

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
    dataset_folder = args.dataset
    json_file = args.dataset + '.json'
    
    # Construct paths
    datasets_dir = '/mlx_devbox/users/cheng.cwang/playground/datasets'
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_flamingo.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)
