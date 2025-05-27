import subprocess
import os
import pickle
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------- 知識庫向量載入與檢索 -----------

def load_kb():
    print("🔄 正在載入知識庫...")
    if not os.path.exists("kb_index.faiss") or not os.path.exists("kb_texts.pkl"):
        print("⚠️ 找不到知識庫檔案，RAG 功能停用")
        return None, None, None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("kb_index.faiss")
    with open("kb_texts.pkl", "rb") as f:
        kb_texts = pickle.load(f)
    print(f"✅ 已載入知識庫，共 {len(kb_texts)} 筆")
    return model, index, kb_texts

kb_model, kb_index, kb_texts = load_kb()

# ----------- 知識庫摘要壓縮 -----------
def summarize_retrieved_kb(retrieved, model="phi4-mini"):
    if not retrieved:
        print("⚠️ 無資料可摘要")
        return ""

    print("🧠 正在產生知識庫摘要...")
    prompt = "請根據以下知識庫內容，整理出一段精簡摘要，幫助我更快理解處理方式與重點：\n\n"
    for i, chunk in enumerate(retrieved, 1):
        prompt += f"{i}. {chunk.strip()}\n"
    prompt += "\n請幫我統整出一段摘要："

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )

        if result.returncode != 0:
            print("⚠️ 壓縮模型錯誤：", result.stderr.decode("utf-8"))
            return ""
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        print(f"⚠️ 壓縮時發生錯誤：{str(e)}")
        return ""

# ----------- 知識庫檢索 -----------
def search_knowledge_base(query, top_k=3):
    print("🔍 執行語意查詢...")
    if kb_model is None or kb_index is None or kb_texts is None:
        print("❌ 知識庫尚未載入")
        return []
    query_vec = kb_model.encode([query])
    D, I = kb_index.search(np.array(query_vec), top_k)
    print(f"[RAG] 🔍 查詢: {query}")
    print(f"[RAG] 🧠 取出知識庫資料：{[kb_texts[i][:50] for i in I[0]]}")
    return [kb_texts[i] for i in I[0]]

# ----------- 類型判斷（查詢 or 統計） -----------
def classify_query_type(message):
    print("🤖 判斷提問類型...")
    system_prompt = (
        "You are a classification assistant. Based on the user's question, determine whether it belongs to one of the following types:"
        "1. Semantic Query (the user wants to find similar past cases or ask how to solve an issue)"
        "2. Statistical Analysis (the user wants to know counts, most frequent categories, or metadata summaries)"
        "Please respond with only one of the following: 'Semantic Query' or 'Statistical Analysis'."
    )
    prompt = f"{system_prompt}\n\n使用者提問：{message}"

    try:
        result = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=30
        )
        reply = result.stdout.decode("utf-8").strip()
        print(f"[分類判斷] 📥 回覆：{reply}")
        return reply
    except Exception as e:
        print(f"⚠️ 分類判斷失敗：{str(e)}")
        return "Semantic Query"

# ----------- 統計分析查詢 -----------
def analyze_metadata_query(message):
    print("📊 執行統計分析...")
    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return f"⚠️ 無法載入 metadata：{str(e)}"

    system_prompt = (
        "You are an assistant helping to analyze a structured knowledge base.\n"
        "Based on the user's message, determine which field should be used for statistical aggregation.\n"
        "You can choose one of: subcategory, configurationItem, roleComponent, or location.\n"
        "Just return the field name, nothing else."
    )
    prompt = f"{system_prompt}\n\nUser: {message}"

    try:
        result = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=30
        )
        field = result.stdout.decode("utf-8").strip()
        print(f"[欄位判斷] 使用欄位：{field}")
        if field not in ["subcategory", "configurationItem", "roleComponent", "location"]:
            return f"⚠️ 無法判斷要統計的欄位（回覆為：{field}）"

        counts = {}
        for item in metadata:
            key = item.get(field, "未標註")
            counts[key] = counts.get(key, 0) + 1

        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        result_lines = [f"{i+1}. {k}：{v} 筆" for i, (k, v) in enumerate(sorted_counts[:5])]
        return f"📊 統計結果（依 {field}）：\n" + "\n".join(result_lines)

    except Exception as e:
        return f"⚠️ 呼叫模型分類欄位時出錯：{str(e)}"

# ----------- GPT 主函式 -----------
def run_offline_gpt(message, model="mistral", history=[]):
    print("🟢 啟動 GPT 回答流程...")
    query_type = classify_query_type(message)
    if query_type == "Statistical Analysis":
        return analyze_metadata_query(message)

    print("🔄 類型為語意查詢，開始檢索知識庫...")
    retrieved = search_knowledge_base(message, top_k=3)
    if retrieved:
        print(f"\n[RAG] ✅ 啟用知識庫輔助，共取出 {len(retrieved)} 筆：")
        for i, chunk in enumerate(retrieved, 1):
            preview = chunk.replace('\n', ' ')[:100]
            print(f"    {i}. {preview}...")
    else:
        print("\n[RAG] ⚠️ 未使用知識庫（未找到相似資料）")

    print(f"[🔧 壓縮用模型] 使用模型：phi4-mini")
    print(f"[🎯 回答用模型] 使用模型：{model}")

    kb_context = summarize_retrieved_kb(retrieved, model="phi4-mini")

    context = ""
    for turn in history[-5:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        context += f"{role}: {turn['content']}\n"

    prompt = f"{kb_context}\n\n{context}User: {message}\nAssistant:"

    print("\n[Prompt Preview] 🧾 發送給模型的 Prompt 前 300 字：")
    print(prompt[:300])

    try:
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
        print("\n📥 模型回覆：", reply)
        return reply if reply else "⚠️ 沒有收到模型回應。"

    except Exception as e:
        return f"⚠️ 呼叫模型時發生錯誤：{str(e)}"
