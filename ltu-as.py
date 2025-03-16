import json
import os
import torch
import time
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import numpy as np
import re
import skimage.measure
import whisper_at
from whisper.model import Whisper, ModelDimensions

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 初始化Whisper模型
def init_whisper_models():
    # 文本转录模型
    whisper_text = whisper_at.load_model("large-v2", device='cuda:1' if device == "cuda" else "cpu")
    
    # 特征提取模型
    checkpoint = torch.load('../../pretrained_mdls/large-v1.pt', 
                          map_location='cuda:0' if device == "cuda" else "cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    whisper_feat = Whisper(dims)
    whisper_feat.load_state_dict(checkpoint["model_state_dict"], strict=False)
    whisper_feat = whisper_feat.to('cuda:0' if device == "cuda" else "cpu")
    
    return whisper_text, whisper_feat

whisper_text_model, whisper_feat_model = init_whisper_models()

# 加载LLaMA基础模型
base_model_path = "../../pretrained_mdls/vicuna_ltuas/"
lora_weights_path = '../../pretrained_mdls/ltuas_long_noqa_a6.bin'

print("Loading base model...")
base_model = LlamaForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)

# 应用LoRA配置
print("Applying LoRA configuration...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, peft_config)

# 加载适配器权重
print("Loading LoRA weights...")
state_dict = torch.load(lora_weights_path, map_location=device)
load_result = model.load_state_dict(state_dict, strict=False)

# 打印加载状态
print(f"Missing keys: {load_result.missing_keys}")
print(f"Unexpected keys: {load_result.unexpected_keys}")

# 设备转移
model = model.to(device)
model.eval()
print("Model loaded successfully")

# 初始化tokenizer
tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token_id = 0

# 文本处理工具
class TextProcessor:
    @staticmethod
    def remove_thanks(text):
        return re.sub(r"(?i)thanks? (you )?for watching!?", "", text).strip()
    
    @staticmethod
    def trim_response(text):
        return text.split("### Response:")[-1].strip()

text_processor = TextProcessor()

# 音频处理模块
class AudioProcessor:
    def __init__(self, text_model, feat_model):
        self.text_cache = {}
        self.text_model = text_model
        self.feat_model = feat_model

    def process_audio(self, filename):
        if filename not in self.text_cache:
            result = self.text_model.transcribe(filename)
            clean_text = text_processor.remove_thanks(result["text"])
            self.text_cache[filename] = clean_text
        
        # 提取音频特征
        _, audio_feat = self.feat_model.transcribe_audio(filename)
        audio_feat = audio_feat[0].permute(2, 0, 1).cpu().numpy()
        audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
        audio_feat = audio_feat[1:]  # 移除首层
        
        return torch.tensor(audio_feat), self.text_cache[filename]

audio_processor = AudioProcessor(whisper_text_model, whisper_feat_model)

# 响应生成配置
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.95,
    top_k=500,
    repetition_penalty=1.1,
    max_new_tokens=500,
    do_sample=True
)

def respond(audio_path, question):
    try:
        start_time = time.time()
        
        # 处理音频输入
        audio_feat, transcript = audio_processor.process_audio(audio_path)
        audio_feat = audio_feat.unsqueeze(0).to(device)
        if device == "cuda":
            audio_feat = audio_feat.half()
        
        # 生成prompt
        prompt = f"""Below is an instruction describing a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{question}

### Input:
{transcript}

### Response:"""
        
        # 生成响应
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                audio_input=audio_feat,
                generation_config=generation_config
            )
        
        # 后处理
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = text_processor.trim_response(full_response)
        
        # 记录日志
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio": audio_path,
            "question": question,
            "response": final_response,
            "processing_time": round(time.time()-start_time, 2)
        }
        
        os.makedirs("./logs", exist_ok=True)
        with open(f"./logs/log_{time.strftime('%Y%m%d')}.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            
        return final_response
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return "Error: Unable to process request at this time"

if __name__ == "__main__":
    # 测试用例
    test_audio = "/home/chengwang/ltu/sample_audio.wav"
    test_question = "What sounds can be heard in this audio?"
    
    print("Running test inference...")
    response = respond(test_audio, test_question)
    print("\nTest Result:")
    print(f"Question: {test_question}")
    print(f"Response: {response}")
