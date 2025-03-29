import subprocess
import argparse
import os
import json
import signal
import sys
from tqdm import tqdm
import time

class SwiftInferSession:
    def __init__(self, cuda_device):
        # Set environment variable for the correct GPU device.
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        # Launch the swift infer process.
        # Note: We assume that swift infer remains running and can accept multiple prompt inputs.
        command = [
            "swift", "infer",
            "--ckpt_dir", "/mlx_devbox/users/cheng.cwang/playground/datasets/output/v6-20250329-064216/checkpoint-100",
            "--load_dataset_config", "true"
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env
        )

        # Optionally, wait a moment for the model to load.
        print(f"Loading model on CUDA device {cuda_device} (this may take a while)...")
        time.sleep(100)  # Adjust delay as needed

    def infer(self, prompt_text, audio_file_path):
        """
        Send inputs to the swift infer process in two separate steps:
        1. First send the prompt with <audio> tag
        2. Then send the audio file path
        With appropriate wait times between operations.
        """
        try:
            # Remove any newlines from the prompt to prevent early input termination
            # and replace them with spaces to preserve meaning
            prompt_text = prompt_text.replace('\n', ' ')
            
            # Format the prompt with <audio> tag if not already included
            if "<audio>" not in prompt_text:
                formatted_prompt = f"<audio>{prompt_text}"
            else:
                formatted_prompt = prompt_text
            
            # Print the first input
            print(f"\nINPUT (1st):")
            print(f"Prompt: {formatted_prompt}")
            
            # Send the first input (prompt)
            self.process.stdin.write(formatted_prompt + "\n")
            self.process.stdin.flush()
            
            # Wait 1 second between inputs
            time.sleep(1)
            
            # Remove any newlines from the audio path
            audio_file_path = audio_file_path.replace('\n', '')
            
            # Print the second input
            print(f"INPUT (2nd):")
            print(f"Audio path: {audio_file_path}")
            
            # Send the second input (audio path)
            self.process.stdin.write(audio_file_path + "\n")
            self.process.stdin.flush()
            
            # Wait 2 seconds for the response
            time.sleep(2)

            # Read response lines until we get the end signal
            full_response = []
            last_content = None
            
            while True:
                line = self.process.stdout.readline().strip()
                
                # Check if this is the end signal
                if line == "no" or line.endswith("----------"):
                    break
                
                # Extract content after the last "<<<" if present
                if "<<<" in line:
                    parts = line.split("<<<")
                    # Get the last part (after the last <<<)
                    last_content = parts[-1].strip()
                else:
                    # If no "<<<", just use the whole line
                    last_content = line
                
                full_response.append(last_content)
            
            # Join all clean response lines
            response = "\n".join(full_response)
            
            # Print the cleaned response
            print(f"CLEANED RESPONSE: {response}")
            
            return response

        except BrokenPipeError:
            print("Error: Broken pipe detected when communicating with the server")
            print("Terminating the entire process...")
            # Force exit the program
            os._exit(1)
        except Exception as e:
            print(f"Error during inference: {e}")
            print("Terminating the entire process...")
            # Force exit the program
            os._exit(1)

    def close(self):
        if self.process:
            print("Shutting down swift infer process...")
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            print("Swift infer process terminated.")


def process_aqa_data(input_json_path, output_json_path, infer_session, dataset_type):
    # Load dataset JSON file.
    with open(input_json_path, 'r') as f:
        aqa_data = json.load(f)

    base_dir = os.path.dirname(input_json_path)
    results = []

    # Apply dataset filtering based on type
    if dataset_type in ['AQA', 'SER']:
        # Skip the first 125 entries as required
        filtered_data = aqa_data[125:]
        print(f"Excluding first 125 entries for {dataset_type} dataset. Processing {len(filtered_data)} entries.")
    else:
        filtered_data = aqa_data
        print(f"Processing all {len(filtered_data)} entries for {dataset_type} dataset.")

    for item in tqdm(filtered_data, desc="Processing audio files"):
        audio_path = os.path.join(base_dir, item['audio'])
        result = {
            'audio': item['audio'],
            'gt': item['gt']
        }
        prompt_types = ['faithful', 'adversarial', 'irrelevant', 'neutral']
        for prompt_type in prompt_types:
            if prompt_type in item:
                # Call the persistent inference session.
                response = infer_session.infer(item[prompt_type], audio_path)
                result[f'{prompt_type}_response'] = response
        results.append(result)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Processing complete. Results saved to {output_json_path}")


# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nCtrl+C detected. Cleaning up...")
    if 'infer_session' in globals():
        globals()['infer_session'].close()
    print("Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Evaluate audio dataset using persistent swift infer session')
    parser.add_argument('--dataset', type=str, required=True, choices=['AQA', 'SER', 'VSC'],
                        help='Dataset folder to process (AQA, SER, or VSC)')
    parser.add_argument('--output_dir', type=str, default='res',
                        help='Directory to save results (default: res)')
    parser.add_argument('--cuda_device', type=int, default=6,
                        help='CUDA device ID to use (default: 6)')
    args = parser.parse_args()

    # Construct input JSON file path.
    dataset_folder = args.dataset
    json_file = f"{dataset_folder}.json"
    datasets_dir = os.path.dirname(os.path.abspath(__file__))
    input_json_path = os.path.join(datasets_dir, dataset_folder, json_file)

    # Construct output JSON file path.
    output_json_path = os.path.join(args.output_dir, f'{dataset_folder.lower()}_swift_infer.json')

    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    print(f"Using CUDA device: {args.cuda_device}")

    # Create a persistent swift infer session.
    try:
        infer_session = SwiftInferSession(args.cuda_device)
        process_aqa_data(input_json_path, output_json_path, infer_session, dataset_folder)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if 'infer_session' in locals():
            infer_session.close()
