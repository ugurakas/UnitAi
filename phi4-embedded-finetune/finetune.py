from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

max_seq_length = 1024

# Unsloth altyapısında sansürsüz Phi-4 çalıştırmak için optimize edilmiş sürüm
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ccharnkij/Phi-4-Uncensored",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True, 
    fix_tokenizer=True, # Tokenizer hatalarını önlemek için kritik
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset("json", data_files="embedded_dataset.jsonl", split="train")

def formatting_func(example):
    output_texts = []
    for ins, out in zip(example["instruction"], example["output"]):
        text = f"### Instruction:\n{ins}\n\n### Response:\n{out}"
        output_texts.append(text)
    return output_texts

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        output_dir="phi4-embedded-lora",
        save_strategy="epoch",
    ),
)

print("Eğitim başlıyor...")
trainer.train()

print("Eğitim bitti, adaptör kaydediliyor...")
model.save_pretrained("phi4-embedded-lora-final")
tokenizer.save_pretrained("phi4-embedded-lora-final")
print("Tamamlandı -> phi4-embedded-lora-final/")
