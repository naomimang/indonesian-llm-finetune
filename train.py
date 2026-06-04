"""
Indonesian LLM Fine-tuning on AMD ROCm
Fine-tuning Llama 3.2 3B for Indonesian language tasks using QLoRA
Requires: AMD GPU with 40GB+ VRAM (MI250/MI300)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on Indonesian dataset")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples from dataset")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("Indonesian LLM Fine-tuning with ROCm")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   PyTorch version: {torch.__version__}")
        if torch.version.hip:
            print(f"   ROCm version: {torch.version.hip}")
    else:
        print("\n❌ No ROCm GPU detected. This script requires AMD GPU with ROCm.")
        print("   Please run on AMD Developer Cloud with MI250 or MI300.")
        return
    
    # Quantization config (4-bit to save memory)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"\n📦 Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model parameters:")
    print(f"   Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"   Total: {total_params:,}")
    
    # Load Indonesian dataset
    print(f"\n📚 Loading Indonesian dataset...")
    try:
        dataset = load_dataset("mesolitica/instruction-22k-indonesian", split=f"train[:{args.max_samples}]")
    except:
        print("   Using fallback dataset...")
        dataset = load_dataset("indonlp/indonlu", split="train[:500]")
    
    def format_prompt(example):
        if "instruction" in example and "output" in example:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        elif "text" in example:
            text = example["text"]
        else:
            text = str(example)
        
        return tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./indonesian-llm-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        eval_strategy="no",
        learning_rate=args.learning_rate,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        save_total_limit=2,
        dataloader_num_workers=4,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("\n🚀 Starting fine-tuning...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size} (gradient accumulation: {args.gradient_accumulation})")
    print(f"   Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Max sequence length: {args.max_length}")
    print(f"\n   Estimated time: 24-48 hours on AMD MI250")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    model.save_pretrained("./indonesian-llm-final")
    tokenizer.save_pretrained("./indonesian-llm-final")
    print("\n✅ Fine-tuning complete!")
    print(f"   Model saved to: ./indonesian-llm-final")
    print("\n📝 Next steps:")
    print("   1. Test inference: python test_inference.py")
    print("   2. Upload to Hugging Face: huggingface-cli upload")
    print("=" * 60)

if __name__ == "__main__":
    main()