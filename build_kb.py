import os
import sys
from tqdm import tqdm

# ✅ 建立 log，確認是否真的有呼叫成功
with open("kb_log.txt", "a", encoding="utf-8") as f:
    f.write("✅ [LOG] build_kb.py 被執行！\n")

print("【DEBUG】你有呼叫到 build_kb.py！")

LOCK_FILE = "kb_building.lock"
if os.path.exists(LOCK_FILE):
    print("❗知識庫正在建立中，請稍候")
    with open("kb_log.txt", "a", encoding="utf-8") as f:
        f.write("❗ [LOG] 偵測到 lock file，已中止建立\n")
    sys.exit(0)

with open(LOCK_FILE, "w") as f:
    f.write("building")

import json
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_DIR = "json_data"
KB_INDEX = "kb_index.faiss"
KB_TEXTS = "kb_texts.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def extract_texts_from_json(json_file):
    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "data" in data:
            items = data["data"]
        else:
            items = [data]
        kb_texts = []
        for item in items:
            problem = item.get("problem") or item.get("問題") or ""
            resolution = item.get("resolution") or item.get("解決方案") or ""
            summary = item.get("summary") or item.get("摘要") or ""
            text = f"問題：{problem}\n處理方式：{resolution}\n摘要：{summary}"
            kb_texts.append(text)
        return kb_texts

def build_kb():
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    all_texts = []

    for file in tqdm(all_files, desc="🔄 掃描 JSON 檔案"):
        path = os.path.join(DATA_DIR, file)
        all_texts.extend(extract_texts_from_json(path))

    for idx, file in enumerate(all_files, 1):
        path = os.path.join(DATA_DIR, file)
        all_texts.extend(extract_texts_from_json(path))
        print(f"已處理 {idx}/{len(all_files)} 個檔案...")

    print(f"共讀取 {len(all_texts)} 筆知識資料，開始建立向量庫...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(all_texts, show_progress_bar=True)
    print("已產生向量，開始寫入 FAISS 檔案...")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, KB_INDEX)

    with open(KB_TEXTS, "wb") as f:
        pickle.dump(all_texts, f)

    with open("kb_log.txt", "a", encoding="utf-8") as f:
        f.write(f"✅ [LOG] 成功建立知識庫，共 {len(all_texts)} 筆\n")

    print("知識庫向量檔已完成")

if __name__ == "__main__":
    try:
        build_kb()
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        print("知識庫建立完成，鎖定檔已刪除")
        with open("kb_log.txt", "a", encoding="utf-8") as f:
            f.write("✅ [LOG] 知識庫流程結束，lock 已清除\n")
