# Running Deepseek locally
This is an attempt to run deep seek r1 model locally on my laptop. 
We are going download the pretrained model DeepSeek-R1-Distill-Qwen-1.5B  using the transformers library by provided by Hugging Face, which would typically download it from the Hugging Face Model Hub.


# Configuration & Prerequisites
- CPU -  Processor	12th Gen Intel(R) Core(TM) i7-12800H, 2400 Mhz, 14 Core(s), 20 Logical Processor(s)
- GPU - NVDIA RTX A2000 8GB
- Physical memory - 32 GB
- Download CUDA and CUDANN from Nvidia that supports your GPU
- Install torch libraries that supports cuda
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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