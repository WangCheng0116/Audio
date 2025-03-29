import torch
torch.cuda.empty_cache()

import librosa
import torch
import argparse
import os
import json
import warnings
from tqdm import tqdm

# Set CUDA device

# Import Swift libraries
from swift.llm import (
    PtEngine, RequestConfig, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process audio with Qwen2 model using Swift')
parser.add_argument('--dataset', type=str, required=True, choices=['AQA', 'SER', 'VSC'], 
                    help='Dataset folder to process (AQA, SER, or VSC)')
parser.add_argument('--output_dir', type=str, default='res', 
                    help='Directory to save results (default: res)')
args = parser.parse_args()

# Model and checkpoint configurations
model_name = 'Qwen/Qwen2-Audio-7B-Instruct'
lora_checkpoint = '/mlx_devbox/users/cheng.cwang/playground/datasets/output/v6-20250329-064216/checkpoint-100'
template_type = None  # Use default template_type for the model
default_system = None  # Use default system prompt for the model

print("Loading model and tokenizer...")
model, tokenizer = get_model_tokenizer(model_name)
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=1)  # Process one by one
request_config = RequestConfig(max_tokens=100, temperature=0)  # Match original max_new_tokens

warnings.filterwarnings("ignore")

def respond(audio_path, input_text):
    """Generate a response using the Swift PtEngine for an audio file and text prompt"""
    try:
        # Create inference request with text message and audio path
        infer_request = InferRequest(
            messages=[{'role': 'user', 'content': "<audio>"+input_text}],
            audios=[audio_path]
        )
        
        # Generate response (process one at a time)
        response = engine.infer([infer_request], request_config)
        
        # Extract and return text response
        result = response[0].choices[0].message.content
        return result
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return f"Error: {str(e)}"

def process_aqa_data(input_json_path, output_json_path):
    """Process dataset and generate responses for each prompt type"""
    with open(input_json_path, 'r') as f:
        aqa_data = json.load(f)
    
    base_dir = os.path.dirname(input_json_path)
    results = []
    
    # Determine start index based on dataset - skip first 125 items for AQA and SER
    start_index = 0
    if args.dataset in ['AQA', 'SER']:
        start_index = 125
        print(f"Skipping first 125 items for {args.dataset} dataset")
    
    # Process items one by one starting from the specified index
    for idx, item in enumerate(tqdm(aqa_data[start_index:], desc="Processing audio files")):
        audio_path = os.path.join(base_dir, item['audio'])
        
        result = {
            'audio': item['audio'],
            'gt': item['gt']
        }
        
        # Process each prompt type if present in the data
        prompt_types = ['faithful', 'adversarial', 'irrelevant', 'neutral']
        for prompt_type in prompt_types:
            if prompt_type in item:
                response = respond(audio_path, item[prompt_type])
                result[f'{prompt_type}_response'] = response
                print(response)
        
        results.append(result)
        
        # Save intermediate results periodically
        if (idx + 1) % 10 == 0:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path + f".partial_{idx+1}", 'w') as f:
                json.dump(results, f, indent=2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Save final results
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
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_qwen2_lora.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)
