import subprocess
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------- 知識庫向量載入與檢索 -----------

# 你可以把這段放到專案啟動時（如 Flask 啟動後就先執行一次）
def load_kb():
    if not os.path.exists("kb_index.faiss") or not os.path.exists("kb_texts.pkl"):
        print("⚠️ 找不到知識庫檔案，RAG 功能停用")
        return None, None, None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("kb_index.faiss")
    with open("kb_texts.pkl", "rb") as f:
        kb_texts = pickle.load(f)
    print(f"已載入知識庫：{len(kb_texts)} 條")
    return model, index, kb_texts

# ⚠️ 注意，這三個變數全域宣告
kb_model, kb_index, kb_texts = load_kb()

def search_knowledge_base(query, top_k=3):
    if kb_model is None or kb_index is None or kb_texts is None:
        return []
    query_vec = kb_model.encode([query])
    D, I = kb_index.search(np.array(query_vec), top_k)
    return [kb_texts[i] for i in I[0]]

# ----------- GPT 主函式 -----------

def run_offline_gpt(message, model="mistral", history=[]):
    """
    使用 Ollama 執行本地 GPT 模型推論，支援知識庫檢索（RAG）與上下文。
    """
    # 🔍 Step 1: 檢索知識庫
    retrieved = search_knowledge_base(message, top_k=3)
    kb_context = "\n".join([f"🔍相關資料：{chunk}" for chunk in retrieved]) if retrieved else ""

    # 🔁 組合歷史對話（最多取 5 筆）
    context = ""
    for turn in history[-5:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        context += f"{role}: {turn['content']}\n"

    # 🧠 整合所有上下文、知識庫與提問
    prompt = f"{kb_context}\n\n{context}User: {message}\nAssistant:"

    # ✅ DEBUG
    print("🔧 使用模型：", model)
    print("📤 傳送 prompt：\n", prompt)

    try:
        # 🛠 呼叫 Ollama CLI
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )

        if result.returncode != 0:
            print("⚠️ Ollama 錯誤：", result.stderr.decode('utf-8'))
            return f"⚠️ Ollama 錯誤：{result.stderr.decode('utf-8')}"

        reply = result.stdout.decode("utf-8").strip()
        print("📥 模型回覆：", reply)
        return reply if reply else "⚠️ 沒有收到模型回應。"

    except Exception as e:
        return f"⚠️ 呼叫模型時發生錯誤：{str(e)}"
