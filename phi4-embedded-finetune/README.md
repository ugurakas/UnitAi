# Phi-4 Embedded Systems Fine-tuning

Phi-4 14B modelini LoRA/QLoRA ile embedded systems (C/C++, STM32, Zephyr, FreeRTOS, ESP-IDF) alanında uzmanlaştırma projesi.

## Yaklaşım

1. Açık kaynaklı embedded repolardan (Zephyr, FreeRTOS, ESP-IDF, STM32Cube, libopencm3) kod parçaları toplandı
2. Self-instruct yöntemiyle (base model kullanılarak) her kod parçası için soru-cevap çiftleri üretildi
3. Üretilen dataset (412 örnek) ile Phi-4 üzerinde LoRA fine-tuning yapıldı
4. Adaptör GGUF formatına çevrilip llama.cpp ile serve edildi

## Kullanım

```bash
pip install -r requirements.txt
python3 generate_dataset.py   # dataset üretimi
python3 finetune.py           # LoRA fine-tuning
```

## Dataset

`embedded_dataset.jsonl` — 412 embedded C/C++ örneği, instruction/output formatında.

## Model

Eğitilmiş LoRA adaptörü ve GGUF: [Hugging Face linkin buraya]
