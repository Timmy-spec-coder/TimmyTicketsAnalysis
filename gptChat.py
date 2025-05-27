import subprocess
import os
import pickle
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64


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





# ----------- 提問類型分類 -----------
def classify_query_type(message):
    print("🤖 判斷提問類型...")
    system_prompt = (
        "You are a classification assistant. Based on the user's question, determine whether it belongs to one of the following types:\n"
        "1. Semantic Query (user wants similar past cases or issue solving)\n"
        "2. Statistical Analysis (user wants counts or summaries)\n"
        "3. Field Filter (user asks to find cases matching a specific field=value)\n"
        "4. Field Values (user wants to know all possible values of a specific field)\n"
        "5. Temporal Trend (user asks about changes over time, trends, or patterns over a date range)\n"
        "6. Solution Summary (user wants a list or summary of known fixes/remedies/resolutions)\n"
        "Please respond with one of: 'Semantic Query', 'Statistical Analysis', 'Field Filter', 'Field Values', 'Temporal Trend', 'Solution Summary'."
    )
    prompt = f"{system_prompt}\n\nUser: {message}"

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
        "You are helping analyze a structured knowledge base.\n"
        "From the user's question, choose ONE of the following fields to do statistical aggregation:\n"
        " - subcategory\n - configurationItem\n - roleComponent\n - location\n"
        "If the request is vague or unclear, respond with '__fallback__'.\n"
        "Only return one word: the field name or '__fallback__'."
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
        print(f"[欄位判斷] GPT 回覆欄位：{field}")

        if field == "__fallback__":
            return "\n\n".join([
                summarize_field("subcategory", metadata),
                summarize_field("configurationItem", metadata)
            ])

        if field not in ["subcategory", "configurationItem", "roleComponent", "location"]:
            return f"⚠️ 無法判斷要統計的欄位（回覆為：{field}）"

        return summarize_field(field, metadata)

    except Exception as e:
        return f"⚠️ 呼叫模型分類欄位時出錯：{str(e)}"

def summarize_field(field, metadata):
    counts = {}
    for item in metadata:
        key = item.get(field, "未標註")
        counts[key] = counts.get(key, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    result_lines = [f"{i+1}. {k}：{v} 筆" for i, (k, v) in enumerate(sorted_counts[:5])]
    return f"📊 統計結果（依 {field}）：\n" + "\n".join(result_lines)

def list_field_values(message):
    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return f"⚠️ Failed to load metadata: {str(e)}"

    # 由 GPT 判斷使用者問的是哪一個欄位
    system_prompt = (
        "You are a parser. The user is asking about what values are available in a certain field.\n"
        "Please extract which field they want to list.\n"
        "Return the field name only. Must be one of: configurationItem, subcategory, roleComponent, location"
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
        if field not in ["configurationItem", "subcategory", "roleComponent", "location"]:
            return f"⚠️ Invalid field: {field}"

        values = set()
        for item in metadata:
            value = item.get(field)
            if value: values.add(value)

        sorted_vals = sorted(values)
        lines = [f"- {v}" for v in sorted_vals[:20]]  # 最多顯示 20 個
        return f"📋 Values in '{field}' field:\n" + "\n".join(lines)

    except Exception as e:
        return f"⚠️ Failed to process: {str(e)}"

    


# ----------- 欄位查詢分析 -----------
def analyze_field_query(message):
    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return f"⚠️ Failed to load metadata: {str(e)}"

    # 🧠 讓 GPT 萃取多個欄位條件
    system_prompt = (
        "You are a parser. Extract all field=value conditions from the user's message for filtering.\n"
        "Only include fields: configurationItem, subcategory, roleComponent, location\n"
        "Return a JSON array like: "
        "[{\"field\": \"subcategory\", \"value\": \"Login/Access\"}, {\"field\": \"location\", \"value\": \"Taipei\"}]"
    )
    prompt = f"{system_prompt}\n\nUser: {message}"

    try:
        result = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=30
        )
        raw = result.stdout.decode("utf-8").strip()
        print("[🔍 多欄位查詢解析]", raw)

        parsed_conditions = json.loads(raw)
        if not isinstance(parsed_conditions, list):
            return "⚠️ Invalid parsed result format."

        # 檢查條件合法性
        allowed_fields = ["configurationItem", "subcategory", "roleComponent", "location"]
        filters = [(c["field"], c["value"]) for c in parsed_conditions if c.get("field") in allowed_fields]

        if not filters:
            return "⚠️ No valid filters extracted from the query."

        # 篩選資料：AND 條件
        def match_all(item):
            return all(item.get(field) == value for field, value in filters)

        matches = [item for item in metadata if match_all(item)]
        if not matches:
            return f"🔍 No results found for: " + " AND ".join([f"{f}={v}" for f, v in filters])

        lines = [f"- {item.get('text', '')[:100]}..." for item in matches[:5]]
        return (
            f"🔎 Top matches for:\n" +
            "\n".join([f"• {f} = {v}" for f, v in filters]) +
            "\n\n" + "\n".join(lines)
        )

    except Exception as e:
        return f"⚠️ Failed to parse or search: {str(e)}"



def analyze_temporal_trend(message):
    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return f"⚠️ 無法載入資料：{str(e)}"

    if not metadata or "analysisTime" not in metadata[0]:
        return "⚠️ 缺少 analysisTime 欄位，無法分析趨勢。"

    # 轉為 DataFrame
    df = pd.DataFrame(metadata)
    df["analysisTime"] = pd.to_datetime(df["analysisTime"], errors="coerce")
    df = df.dropna(subset=["analysisTime"])

    # 群組每月數量
    df["month"] = df["analysisTime"].dt.to_period("M")
    trend = df.groupby("month").size()

    # 畫圖
    plt.figure(figsize=(8, 4))
    trend.plot(kind="line", marker="o")
    plt.title("📈 趨勢圖：每月案件數")
    plt.xlabel("月份")
    plt.ylabel("案件數量")
    plt.tight_layout()

    # 轉為 base64 圖片嵌入
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    return f"<img src='data:image/png;base64,{img_base64}' alt='Trend chart'>"


# 存入記憶（以最後一個 query 為基礎）
def save_query_context(chat_id, query, result_type, filter_info=None, result_summary=None):
    filepath = f"chat_history/{chat_id}.json"
    try:
        with open(filepath, encoding="utf-8") as f:
            history = json.load(f)
    except:
        history = []

    # 🔁 更新最新記憶
    context = {
        "type": result_type,
        "query": query,
        "filters": filter_info,          # e.g. {"field": "subcategory", "value": "Crash/Hang"}
        "summary": result_summary        # 可選：簡化摘要
    }

    if history:
        history[-1]["context"] = context

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)



def is_follow_up_query(message: str) -> bool:
    keywords = ["previous", "last query", "those", "add filter", "now show", "continue", "follow up"]
    return any(kw in message.lower() for kw in keywords)



def handle_follow_up(chat_id, message):
    filepath = f"chat_history/{chat_id}.json"
    try:
        with open(filepath, encoding="utf-8") as f:
            history = json.load(f)
    except:
        return "⚠️ 無法讀取先前對話記錄，請確認 chat_id 是否正確。"

    if not history or "context" not in history[-1]:
        return "⚠️ 查無先前查詢條件，請重新描述您的需求。"

    context = history[-1]["context"]
    result_type = context.get("type")

    # 👇 根據先前查詢類型接續處理
    if result_type == "Field Filter":
        original = context.get("filters", {})
        new_filter_prompt = (
            "You are a filter parser. Based on this message, extract an additional field and value to add as a filter.\n"
            "Return JSON like: {\"field\": \"subcategory\", \"value\": \"Crash\"}"
        )
        prompt = f"{new_filter_prompt}\n\nUser: {message}"

        try:
            result = subprocess.run(
                ["ollama", "run", "phi3:mini"],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=30
            )
            new_filter = json.loads(result.stdout.decode("utf-8").strip())
            field = new_filter.get("field")
            value = new_filter.get("value")

            if field not in ["configurationItem", "subcategory", "roleComponent", "location"]:
                return "⚠️ 無效的欄位"

            filters = [original] + [new_filter]
            with open("kb_metadata.json", encoding="utf-8") as f:
                metadata = json.load(f)
            matches = metadata
            for f in filters:
                matches = [m for m in matches if m.get(f["field"]) == f["value"]]
            lines = [f"- {item.get('text', '')[:100]}..." for item in matches[:5]]
            return f"🔎 延伸查詢結果（共 {len(matches)} 筆）：\n" + "\n".join(lines)

        except Exception as e:
            return f"⚠️ 延伸查詢失敗：{str(e)}"

    # 你也可以在這裡支援 Statistical Analysis、Semantic Query 等的延伸策略
    return "⚠️ 目前只支援欄位篩選的延伸查詢。"


def summarize_solutions(message):
    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        return f"⚠️ Failed to load metadata: {str(e)}"

    # 搜尋語意相似的案例
    related_cases = search_knowledge_base(message, top_k=5)
    if not related_cases:
        return "⚠️ No similar cases found to extract solutions."

    # 將所有 solution 萃取出來組合
    solutions = []
    for item in metadata:
        text = item.get("text", "")
        if any(snippet in text for snippet in related_cases):
            solution = item.get("solution", "")
            if solution: solutions.append(solution)

    if not solutions:
        return "⚠️ No resolution data found for related cases."

    # 使用 GPT 進行統整
    prompt = "Please summarize the following resolution steps into a brief, clear list:\n\n"
    prompt += "\n---\n".join(solutions[:10])
    prompt += "\n\nSummary:"

    try:
        result = subprocess.run(
            ["ollama", "run", "phi4"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )
        return result.stdout.decode("utf-8").strip()

    except Exception as e:
        return f"⚠️ Failed to summarize solutions: {str(e)}"






# ----------- GPT 主函式 -----------
def run_offline_gpt(message, model="mistral", history=[], chat_id=None):
    print("🟢 啟動 GPT 回答流程...")

    query_type = classify_query_type(message)

    if is_follow_up_query(message) and chat_id:
        return handle_follow_up(chat_id, message)
    print(f"🔍 判斷結果：{query_type}")


    if query_type == "Statistical Analysis":
        reply = analyze_metadata_query(message)
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply

    if query_type == "Field Filter":
        print("🔄 類型為欄位過濾，開始進行過濾...")
        reply = analyze_field_query(message)
        try:
            parsed = json.loads(reply)
            filters = {"field": parsed.get("field"), "value": parsed.get("value")}
        except:
            filters = None
        save_query_context(chat_id, message, query_type, filter_info=filters, result_summary=reply[:200])
        return reply

    if query_type == "Field Values":
        print("📋 類型為欄位值清單，開始列出欄位值...")
        reply = list_field_values(message)
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply

    if query_type == "Temporal Trend":
        print("📈 類型為時間趨勢查詢，開始繪製圖表...")
        reply = analyze_temporal_trend(message)
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply
    
    if query_type == "Solution Summary":
        print("🛠 類型為解法統整，開始彙整處理方式...")
        reply = summarize_solutions(message)
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply



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

        save_query_context(chat_id, message, query_type, result_summary=reply[:200])

        return reply if reply else "⚠️ 沒有收到模型回應。"

    except Exception as e:
        return f"⚠️ 呼叫模型時發生錯誤：{str(e)}"