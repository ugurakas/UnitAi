import requests
import json
import glob
import random
import os

SOURCE_DIR = os.path.expanduser("~/embedded-repos")
LLAMA_SERVER_URL = "http://localhost:8080/v1/chat/completions"

def find_code_files(root_dir, extensions=(".c", ".h"), limit=None):
    files = []
    for ext in extensions:
        files.extend(glob.glob(f"{root_dir}/**/*{ext}", recursive=True))
    random.shuffle(files)
    return files[:limit] if limit else files

def generate_qa(code_snippet):
    prompt = f"""Aşağıdaki embedded C/C++ kod parçasını incele. Bir embedded systems mühendisinin sorabileceği türde 1 teknik soru ve buna kod parçasına dayanan detaylı, doğru bir cevap üret.

SADECE geçerli JSON formatında cevap ver, başka hiçbir şey yazma:
{{"instruction": "...", "output": "..."}}

Kod:
{code_snippet}
"""
    try:
        resp = requests.post(
            LLAMA_SERVER_URL,
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=90
        )
        content = resp.json()["choices"][0]["message"]["content"]
        content = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(content)
    except Exception as e:
        print(f"  [HATA] {e}")
        return None

def main():
    files = find_code_files(SOURCE_DIR, limit=1000)
    print(f"{len(files)} dosya bulundu, işleniyor...")

    dataset = []
    for i, f in enumerate(files):
        try:
            with open(f, errors="ignore") as fp:
                code = fp.read()
        except Exception:
            continue

        if len(code) < 200 or len(code) > 3000:
            continue

        print(f"[{i+1}/{len(files)}] {f}")
        qa = generate_qa(code)
        if qa and "instruction" in qa and "output" in qa:
            dataset.append(qa)
            print(f"  ✓ Soru: {qa['instruction'][:80]}...")

            if len(dataset) % 50 == 0:
                with open("embedded_dataset.jsonl", "w", encoding="utf-8") as out:
                    for row in dataset:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  [ara kayıt: {len(dataset)} örnek]")

    with open("embedded_dataset.jsonl", "w", encoding="utf-8") as out:
        for row in dataset:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nToplam {len(dataset)} örnek üretildi -> embedded_dataset.jsonl")

if __name__ == "__main__":
    main()
