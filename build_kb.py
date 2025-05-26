import os
import sys
import json
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

print("✅ [DEBUG] 你有成功呼叫 build_kb.py")

# ========== ✅ 加入 log 與鎖定檢查 ==========

LOCK_FILE = "kb_building.lock"
LOG_FILE = "kb_log.txt"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log("✅ [LOG] build_kb.py 被執行！")

if os.path.exists(LOCK_FILE):
    print("❗知識庫正在建立中，請稍候")
    log("❗ [LOG] 偵測到 lock file，已中止建立")
    sys.exit(0)

with open(LOCK_FILE, "w") as f:
    f.write("building")

# =============================================

DATA_DIR = "json_data"
KB_INDEX = "kb_index.faiss"
KB_TEXTS = "kb_texts.pkl"
PROCESSED_LOG = "processed_files.json"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_processed_files():
    if not os.path.exists(PROCESSED_LOG):
        return set()
    with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
        data = json.load(f)
        return set(entry["file"] for entry in data)

def save_processed_file(file):
    now = datetime.now().isoformat()
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append({ "file": file, "processedAt": now })

    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
            ai_summary = item.get("aiSummary") or item.get("problemSummary") or "(AI 擷取失敗)"
            solution = item.get("solution") or "(AI 擷取失敗)"
            text = f"問題：{ai_summary}\n處理方式：{solution}"
            kb_texts.append(text)
        return kb_texts

def build_kb():
    processed_files = load_processed_files()
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json") and f not in processed_files]
    if not all_files:
        print("📭 沒有新檔案，跳過建庫")
        return

    print(f"📂 有 {len(all_files)} 個新 JSON 檔要加入知識庫")
    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(KB_INDEX) and os.path.exists(KB_TEXTS):
        print("🔄 載入舊有 FAISS index 與文字庫")
        index = faiss.read_index(KB_INDEX)
        with open(KB_TEXTS, "rb") as f:
            kb_texts = pickle.load(f)
    else:
        print("🆕 建立全新知識庫")
        index = None
        kb_texts = []

    for file in tqdm(all_files, desc="📥 加入新知識檔案"):
        path = os.path.join(DATA_DIR, file)
        new_texts = extract_texts_from_json(path)
        new_embeddings = model.encode(new_texts, show_progress_bar=False)
        if index is None:
            index = faiss.IndexFlatL2(new_embeddings.shape[1])
        index.add(np.array(new_embeddings))
        kb_texts.extend(new_texts)
        save_processed_file(file)

    faiss.write_index(index, KB_INDEX)
    with open(KB_TEXTS, "wb") as f:
        pickle.dump(kb_texts, f)

    print(f"✅ 知識庫更新完成（總共 {len(kb_texts)} 筆）")
    log(f"✅ [LOG] 成功建立知識庫，共 {len(kb_texts)} 筆")

if __name__ == "__main__":
    try:
        build_kb()
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
        print("🗂️ 鎖定檔已刪除，結束建庫流程")
        log("✅ [LOG] 知識庫流程結束，lock 已清除")
