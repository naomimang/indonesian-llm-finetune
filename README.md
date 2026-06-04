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
