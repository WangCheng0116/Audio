from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model = Qwen2AudioForConditionalGeneration.from_pretrained("SeaLLMs/SeaLLMs-Audio-7B", device_map="auto")
processor = AutoProcessor.from_pretrained("SeaLLMs/SeaLLMs-Audio-7B")

def respond(audio_path, prompt):
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audio = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)[0]
    
    inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True, sampling_rate=16000)
    inputs.input_ids = inputs.input_ids.to("cuda")
    inputs = {k: v.to("cuda") for k, v in inputs.items() if v is not None}
    
    generate_ids = model.generate(**inputs, max_new_tokens=128, temperature=0, do_sample=False)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return response


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
                print(response)
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
    datasets_dir = '/home/chengwang/datasets'
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_sea.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)
