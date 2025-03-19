
from model import SALMONN
import torch
import argparse
import os
import json
from tqdm import tqdm

# Hardcoded variables
CKPT_PATH = "/mlx_devbox/users/cheng.cwang/playground/SALMONN-7B/ckpt/salmonn_7b_v0.pth"
WHISPER_PATH = "openai/whisper-large-v2"
BEATS_PATH = "/mlx_devbox/users/cheng.cwang/playground/SALMONN/models/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
VICUNA_PATH = "lmsys/vicuna-7b-v1.5"

# Global variable for the model
model = None

# Initialize the model
def initialize_model(device="cuda:0", low_resource=False, lora_alpha=32):
    global model
    model = SALMONN(
        ckpt=CKPT_PATH,
        whisper_path=WHISPER_PATH,
        beats_path=BEATS_PATH,
        vicuna_path=VICUNA_PATH,
        lora_alpha=lora_alpha,
        low_resource=low_resource
    )
    model.to(device)
    model.eval()
    return model

# Main response function
def respond(audio_path, prompt, num_beams=1, temperature=0.1, top_p=0.9):
    global model
    if model is None:
        raise ValueError("Model not initialized. Call initialize_model() first.")
    
    response = model.generate(
        wav_path=audio_path,
        prompt=prompt,
        num_beams=num_beams,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Return first response (assuming model.generate returns a list)
    return response[0] if isinstance(response, list) else response

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
                print(response)
                result[f'{prompt_type}_response'] = response
        results.append(result)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processing complete. Results saved to {output_json_path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process audio with SALMONN model')
    parser.add_argument('--dataset', type=str, required=True, choices=['AQA', 'SER', 'VSC'], 
                        help='Dataset folder to process (AQA, SER, or VSC)')
    parser.add_argument('--output_dir', type=str, default='res', 
                        help='Directory to save results (default: res)')
    args = parser.parse_args()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Initialize the model once
    initialize_model()
    
    dataset_folder = args.dataset
    json_file = args.dataset + '.json'
    
    # Construct paths
    datasets_dir = '/mlx_devbox/users/cheng.cwang/playground/datasets'
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_salmonn7b.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)
