# Running Deepseek locally
This is an attempt to run the DeepSeek R1 model locally on my laptop. Let's greet DeepSeek and get a response from it. We will download the pretrained DeepSeek-R1-Distill-Qwen-1.5B model using the Transformers library provided by Hugging Face, which will typically fetch it from the Hugging Face Model Hub
# Configuration 
- CPU: 12th Gen Intel(R) Core(TM) i7-12800H, 2400 MHz, 14 cores, 20 logical processors 
- GPU: NVIDIA RTX A2000 8GB 
- Physical Memory: 32 GB
- Download the appropriate versions of CUDA and cuDNN from NVIDIA to ensure compatibility with your GPU
# Github Repo
- https://github.com/balajich/hello-deep-seek.git
# Prerequisites
```bash
cd C:\deep-seek
set PYTHONHOME=C:\soft\python-3.9.13
set PATH=%PYTHONHOME%;%PYTHONHOME%\Scripts;%PATH%
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
# Validate GPU support
```bash
(venv) C:\deep-seek>python gpu_support.py
True
```
# Say Hello to DeepSeek
```bash
(venv) C:\deep-seek>python hello_deep_seek.py
Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
Hello DeepSeek, I'm trying to understand the concept of the "K" in the KNN algorithm. Could you explain it to me?
Certainly! The K in KNN stands for "K Nearest Neighbors." It's a type
```