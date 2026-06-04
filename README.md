# Indonesian LLM Fine-tuning on AMD ROCm

[![ROCm](https://img.shields.io/badge/ROCm-7.0-red)](https://www.amd.com/en/developer/rocm.html)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

## 📌 Overview

This project fine-tunes **Llama 3.2 (3B)** for **Indonesian language tasks** using **QLoRA** (Quantized Low-Rank Adaptation) on **AMD GPUs with ROCm**.

Indonesian is a low-resource language in the LLM space. Most open-source models are optimized for English, Chinese, or European languages. This project aims to bridge that gap by creating a model that understands and generates Indonesian text fluently.

## 🎯 Why AMD ROCm?

| Requirement | Value |
|-------------|-------|
| Minimum VRAM | 40 GB |
| Recommended GPU | AMD MI250 (64GB) or MI300 (128GB) |
| Training time | 24-48 hours |
| Framework | PyTorch with ROCm 7.0 |

**Why not NVIDIA?**  
- NVIDIA consumer GPUs (RTX 4090) only have 24GB VRAM → insufficient for full fine-tuning  
- Cloud NVIDIA A100/H100 are expensive ($3-5/hour)  
- AMD Developer Cloud provides access to MI250 at lower cost for researchers

## 🏗️ Project Structure
indonesian-llm-finetune/
├── train.py # Main training script with QLoRA
├── requirements.txt # Python dependencies
├── README.md # This file
└── LICENSE # MIT License

text

## 🔧 Requirements

### Hardware (on AMD Developer Cloud)
- AMD GPU with ROCm support (MI250 or MI300 recommended)
- 40GB+ VRAM
- 4+ CPU cores
- 50GB+ storage

### Software
- ROCm 7.0+
- Python 3.10+
- PyTorch with ROCm support

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/indonesian-llm-finetune.git
cd indonesian-llm-finetune
2. Install dependencies
bash
pip install -r requirements.txt
3. Run training
bash
python train.py --epochs 3 --max_samples 1000
📊 Training Configuration
Parameter	Value
Base Model	Meta Llama 3.2 3B
Quantization	4-bit (NF4)
LoRA Rank	16
LoRA Alpha	32
Target Modules	q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
Batch Size	1 (gradient accumulation: 4)
Effective Batch Size	4
Learning Rate	2e-4
Epochs	3
Max Sequence Length	512
Optimizer	paged_adamw_8bit
📈 Expected Output
text
============================================================
Indonesian LLM Fine-tuning with ROCm
============================================================

✅ GPU detected: AMD Instinct MI250X
   VRAM: 64.00 GB
   PyTorch version: 2.1.0
   ROCm version: 7.0.0

📦 Loading model: meta-llama/Llama-3.2-3B

📊 Model parameters:
   Trainable: 8,388,608 (0.28%)
   Total: 3,005,248,512

📚 Loading Indonesian dataset...

🚀 Starting fine-tuning...
   Epochs: 3
   Batch size: 1 (gradient accumulation: 4)
   Effective batch size: 4
   Learning rate: 0.0002
   Max sequence length: 512

   Estimated time: 24-48 hours on AMD MI250
============================================================

Epoch 1/3: 100%|██████████| 250/250 [08:15:23]
Epoch 2/3: 100%|██████████| 250/250 [07:58:45]
Epoch 3/3: 100%|██████████| 250/250 [07:42:11]

✅ Fine-tuning complete!
   Model saved to: ./indonesian-llm-final
🔬 Use Cases for Fine-tuned Model
Indonesian chatbots and customer service

Bahasa Indonesia text summarization

Educational tools for Indonesian students

Translation between Indonesian and regional languages (Javanese, Sundanese, Balinese)

Content moderation for Indonesian social media

Indonesian news article generation

Legal document analysis for Indonesian courts

📝 Dataset
This project uses the Mesolitica Indonesian Instruction Dataset (mesolitica/instruction-22k-indonesian), which contains 22,000 instruction-response pairs in Indonesian. The dataset includes:

General knowledge Q&A

Creative writing tasks

Reasoning problems

Translation exercises

Code generation in Indonesian context

Dataset Format
text
### Instruction:
Apa ibu kota Indonesia?

### Response:
Ibu kota Indonesia adalah Jakarta. Namun, pada tahun 2024, Indonesia akan memindahkan ibu kotanya ke Nusantara di Kalimantan Timur.
🧠 Technical Details
QLoRA (Quantized Low-Rank Adaptation)
QLoRA reduces memory usage by:

Quantizing the base model to 4-bit (NF4 format)

Freezing base model weights

Training small LoRA adapters (0.1% of parameters)

This allows fine-tuning a 3B model on a single 40GB GPU instead of requiring 80GB+.

ROCm Optimization
The training script automatically:

Detects available AMD GPUs

Uses ROCm-optimized PyTorch kernels

Enables mixed precision (FP16) for faster training

Supports multi-GPU training via PyTorch DDP

Memory Usage Breakdown
Component	Memory
4-bit Base Model	~1.5 GB
LoRA Adapters	~0.5 GB
Gradients	~1.0 GB
Optimizer States	~2.0 GB
Activations	~2.0 GB
Total	~7 GB
Actual usage may vary based on sequence length and batch size.

🤝 Contributing
Contributions are welcome! Please open an issue or pull request for:

Additional Indonesian datasets

Better training configurations

Evaluation benchmarks for Indonesian LLMs

Support for larger models (Llama 7B, 13B)

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgements
AMD ROCm for GPU computing platform

Meta Llama 3.2 for base model

Hugging Face for transformers and datasets

Mesolitica for Indonesian instruction dataset

Microsoft LoRA for QLoRA paper

📧 Contact
For questions about this project, please open a GitHub issue.
