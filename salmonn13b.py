import argparse

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample


def init_model(cfg_path="/mlx_devbox/users/cheng.cwang/playground/SALMONN/configs/decode_config.yaml", 
               device="cuda:0", 
               options=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, help='path to configuration file', default=cfg_path)
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
        default=options
    )
    
    args = parser.parse_args([])  # Empty list to avoid reading from command line
    if cfg_path:
        args.cfg_path = cfg_path
    if device:
        args.device = device
    if options:
        args.options = options
    
    cfg = Config(args)
    
    # Load model
    print("Loading SALMONN model...")
    model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()
    
    # Load audio processor
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    
    return model, cfg, wav_processor, args


model, cfg, wav_processor, args = init_model()


def respond(audio_path, prompt):

    samples = prepare_one_sample(audio_path, wav_processor)
    
    # Format the prompt according to the model's template
    formatted_prompt = [
        cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
    ]
    
    # Generate the response
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(samples, cfg.config.generate, prompts=formatted_prompt)[0]
    
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
                result[prompt_type + '_response'] = response
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
    output_json_path = os.path.join(args.output_dir, dataset_folder.lower() + '_salmonn.json')
    
    print(f"Processing dataset: {dataset_folder}")
    print(f"Input JSON: {input_json_path}")
    print(f"Output will be saved to: {output_json_path}")
    
    process_aqa_data(input_json_path, output_json_path)


# accelerate                    0.23.0
# aiofiles                      22.1.0
# aiohttp                       3.8.5
# aiosignal                     1.3.2
# aiosqlite                     0.20.0
# alabaster                     0.7.16
# altair                        5.1.2
# annotated-types               0.7.0
# antlr4-python3-runtime        4.9.3
# anyio                         3.7.1
# appdirs                       1.4.4
# argon2-cffi                   23.1.0
# argon2-cffi-bindings          21.2.0
# arrow                         1.3.0
# asttokens                     2.4.1
# async-timeout                 4.0.3
# attrs                         24.2.0
# audioread                     3.0.1
# babel                         2.16.0
# beautifulsoup4                4.12.3
# bleach                        6.2.0
# byted_mario_collector         2.0.8
# byted_remote_ikernel          0.4.8
# byted-torch                   2.1.0.post11
# byted-wandb                   0.13.84
# bytedance-context             0.7.1
# bytedance-metrics             0.5.2
# bytedbackgrounds              0.0.6
# byteddatabus                  1.0.6
# byteddps                      0.1.2
# bytedenv                      0.6.4
# bytedes                       1.0.29
# bytedlogger                   0.15.2
# bytedlogid                    0.2.1
# bytedmemfd                    0.2
# bytedmetrics                  0.10.2
# bytedservicediscovery         0.18.0
# bytedtcc                      1.4.5
# bytedtrace                    0.3.0
# bytedztijwthelper             0.0.23
# bytedztispiffe                0.0.14
# certifi                       2024.8.30
# cffi                          1.17.1
# chardet                       4.0.0
# charset-normalizer            3.3.2
# click                         8.1.7
# cmake                         3.31.6
# comm                          0.2.2
# contourpy                     1.3.0
# cryptography                  42.0.8
# cycler                        0.12.1
# Cython                        3.0.12
# datasets                      2.14.5
# dbus-python                   1.2.16
# debugpy                       1.8.8
# decorator                     5.1.1
# defusedxml                    0.7.1
# Deprecated                    1.2.14
# devscripts                    2.21.3+deb11u1
# dill                          0.3.7
# distro-info                   1.0+deb11u1
# dnspython                     2.7.0
# docker-pycreds                0.4.0
# docutils                      0.19
# entrypoints                   0.4
# enum34                        1.1.10
# exceptiongroup                1.2.2
# executing                     2.1.0
# extension_jupyter_kernel      0.1.0
# fastapi                       0.103.2
# fastjsonschema                2.20.0
# ffmpeg-python                 0.2.0
# ffmpy                         0.5.0
# filelock                      3.15.4
# findspark                     2.0.1
# fire                          0.5.0
# fonttools                     4.43.1
# fqdn                          1.5.1
# frozenlist                    1.5.0
# fsspec                        2023.6.0
# future                        0.18.3
# gitdb                         4.0.11
# GitPython                     3.1.37
# gpg                           1.14.0-unknown
# gradio                        3.47.1
# gradio_client                 0.6.0
# greenlet                      3.1.1
# grpcio                        1.67.1
# h11                           0.14.0
# httpcore                      0.18.0
# httpx                         0.25.0
# huggingface-hub               0.17.3
# idna                          3.8
# imageio                       2.31.5
# imagesize                     1.4.1
# importlib_metadata            8.4.0
# importlib_resources           6.5.2
# iotop                         0.6
# ipaddress                     1.0.23
# ipykernel                     6.29.5
# ipython                       8.18.1
# ipython-genutils              0.2.0
# ipywidgets                    8.1.5
# isoduration                   20.11.0
# jedi                          0.19.2
# Jinja2                        3.1.2
# joblib                        1.4.2
# json5                         0.9.28
# jsonpointer                   3.0.0
# jsonschema                    4.19.1
# jsonschema-specifications     2024.10.1
# jupyter                       1.0.0
# jupyter_client                7.4.9
# jupyter-console               6.6.3
# jupyter_core                  5.7.2
# jupyter-events                0.10.0
# jupyter-kernel-gateway        2.5.2
# jupyter_server                2.14.2
# jupyter_server_fileid         0.9.3
# jupyter_server_terminals      0.5.3
# jupyter_server_ydoc           0.8.0
# jupyter-ydoc                  0.2.5
# jupyterlab                    3.6.8
# jupyterlab_pygments           0.3.0
# jupyterlab_server             2.27.3
# jupyterlab_widgets            3.0.13
# kiwisolver                    1.4.7
# lazy_loader                   0.4
# librosa                       0.11.0
# lit                           18.1.8
# llvmlite                      0.41.1
# MarkupSafe                    2.1.5
# matplotlib                    3.8.0
# matplotlib-inline             0.1.7
# merlin_kernel                 0.1
# mistune                       3.0.2
# mlx-python-sdk                0.3.0
# more-itertools                10.6.0
# mpmath                        1.3.0
# msgpack                       1.0.8
# multidict                     6.2.0
# multiprocess                  0.70.15
# nbclassic                     1.1.0
# nbclient                      0.10.0
# nbconvert                     7.16.4
# nbformat                      5.10.4
# nest-asyncio                  1.6.0
# networkx                      3.2.1
# none                          0.1.1
# notebook                      6.5.7
# notebook_shim                 0.2.4
# numba                         0.58.0
# numpy                         1.23.5
# nvidia-cublas-cu11            11.10.3.66
# nvidia-cublas-cu12            12.4.5.8
# nvidia-cuda-cupti-cu12        12.4.127
# nvidia-cuda-nvrtc-cu11        11.7.99
# nvidia-cuda-nvrtc-cu12        12.4.127
# nvidia-cuda-runtime-cu11      11.7.99
# nvidia-cuda-runtime-cu12      12.4.127
# nvidia-cudnn-cu11             8.5.0.96
# nvidia-cudnn-cu12             9.1.0.70
# nvidia-cufft-cu12             11.2.1.3
# nvidia-curand-cu12            10.3.5.147
# nvidia-cusolver-cu12          11.6.1.9
# nvidia-cusparse-cu12          12.3.1.170
# nvidia-cusparselt-cu12        0.6.2
# nvidia-nccl-cu12              2.21.5
# nvidia-nvjitlink-cu12         12.4.127
# nvidia-nvtx-cu12              12.4.127
# omegaconf                     2.3.0
# openai-whisper                20240930
# orjson                        3.9.7
# overrides                     7.7.0
# packaging                     24.1
# pandas                        2.1.1
# pandocfilters                 1.5.1
# parso                         0.8.4
# pathtools                     0.1.2
# peft                          0.3.0.dev0     /mlx_devbox/users/cheng.cwang/playground/ltu/src/ltu_as/peft-main
# pexpect                       4.8.0
# pillow                        10.2.0
# pip                           24.2
# platformdirs                  4.3.6
# ply                           3.11
# pooch                         1.8.2
# prometheus_client             0.21.0
# promise                       2.3
# prompt_toolkit                3.0.48
# propcache                     0.3.0
# protobuf                      4.24.4
# psutil                        5.9.5
# ptyprocess                    0.7.0
# pure_eval                     0.2.3
# py4j                          0.10.9.7
# pyarrow                       13.0.0
# pycparser                     2.22
# pycurl                        7.43.0.6
# pydantic                      2.4.2
# pydantic_core                 2.10.1
# pydub                         0.25.1
# Pygments                      2.18.0
# PyGObject                     3.38.0
# PyJWT                         2.9.0
# pyOpenSSL                     24.2.1
# pyparsing                     3.2.1
# python-apt                    2.2.1
# python-dateutil               2.9.0.post0
# python-debian                 0.1.39
# python-dotenv                 1.0.1
# python-etcd                   0.4.5
# python-json-logger            2.0.7
# python-magic                  0.4.20
# python-multipart              0.0.20
# pytz                          2025.1
# pyxdg                         0.27
# PyYAML                        6.0.1
# pyzmq                         26.2.0
# qtconsole                     5.6.1
# QtPy                          2.4.3
# reactivex                     4.0.4
# referencing                   0.35.1
# regex                         2023.10.3
# requests                      2.32.3
# rfc3339-validator             0.1.4
# rfc3986                       2.0.0
# rfc3986-validator             0.1.1
# rpds-py                       0.21.0
# safetensors                   0.5.3
# schedule                      1.2.2
# scikit-image                  0.22.0
# scikit-learn                  1.6.1
# scipy                         1.11.3
# semantic-version              2.10.0
# Send2Trash                    1.8.3
# sentencepiece                 0.1.99
# sentry-sdk                    1.31.0
# setproctitle                  1.3.3
# setuptools                    74.0.0
# shortuuid                     1.0.13
# six                           1.16.0
# smmap                         5.0.1
# sniffio                       1.3.0
# snowballstemmer               2.2.0
# soundfile                     0.13.1
# soupsieve                     2.6
# soxr                          0.5.0.post1
# Sphinx                        5.3.0
# sphinxcontrib-applehelp       2.0.0
# sphinxcontrib-devhelp         2.0.0
# sphinxcontrib-htmlhelp        2.1.0
# sphinxcontrib-jsmath          1.0.1
# sphinxcontrib-qthelp          2.0.0
# sphinxcontrib-serializinghtml 2.0.0
# sphinxcontrib-websupport      2.0.0
# SQLAlchemy                    2.0.27
# stack-data                    0.6.3
# starlette                     0.27.0
# sympy                         1.13.1
# termcolor                     2.5.0
# terminado                     0.18.1
# threadpoolctl                 3.6.0
# thriftpy2                     0.5.2
# tifffile                      2023.9.26
# tiktoken                      0.3.3
# timm                          0.4.5
# tinycss2                      1.4.0
# tokenizers                    0.13.3
# tomli                         2.1.0
# toolz                         1.0.0
# torch                         1.13.1
# torchaudio                    0.13.1
# torchvision                   0.14.1
# tornado                       6.4.1
# tqdm                          4.67.1
# traitlets                     5.14.3
# transformers                  4.28.0.dev0    /mlx_devbox/users/cheng.cwang/playground/ltu/src/ltu_as/hf-dev/transformers-main
# triton                        2.0.0
# types-python-dateutil         2.9.0.20241003
# typing_extensions             4.12.2
# tzdata                        2025.1
# ujson                         5.10.0
# unattended-upgrades           0.1
# unidiff                       0.5.5
# uri-template                  1.3.0
# urllib3                       1.26.20
# uvicorn                       0.23.2
# wandb                         0.15.12
# wcwidth                       0.2.13
# webcolors                     24.11.1
# webencodings                  0.5.1
# websocket-client              1.8.0
# websockets                    11.0.3
# wheel                         0.44.0
# whisper-at                    0.5
# widgetsnbextension            4.0.13
# wrapt                         1.16.0
# xdg                           5
# xxhash                        3.5.0
# y-py                          0.6.2
# yarl                          1.18.3
# ypy-websocket                 0.8.4
# zipp                          3.20.1
