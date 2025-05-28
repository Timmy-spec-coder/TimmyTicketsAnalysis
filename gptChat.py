import subprocess
import os
import pickle
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import re
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
import subprocess

def summarize_retrieved_kb(retrieved, model="phi4-mini"):
    if not retrieved:
        print("⚠️ 無資料可摘要（retrieved 為空）")
        return ""

    print("🧠 正在產生知識庫摘要...")
    print(f"📦 輸入資料筆數：{len(retrieved)}")
    
    prompt = "請根據以下知識庫內容，整理出一段精簡摘要，幫助我更快理解處理方式與重點：\n\n"
    for i, chunk in enumerate(retrieved, 1):
        clean_chunk = chunk.strip()
        prompt += f"{i}. {clean_chunk}\n"
        print(f"📄 第 {i} 筆內容：{clean_chunk[:60]}{'...' if len(clean_chunk) > 60 else ''}")

    prompt += "\n請幫我統整出一段摘要："
    print(f"📝 組裝完成的 Prompt（前 300 字）：\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}")

    try:
        print(f"🚀 呼叫模型：{model}，請稍候...")
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=600
        )

        print(f"🔚 模型執行完成，返回碼：{result.returncode}")
        if result.returncode != 0:
            stderr_output = result.stderr.decode("utf-8")
            print("⚠️ 壓縮模型錯誤訊息：", stderr_output)
            return ""

        summary = result.stdout.decode("utf-8").strip()
        print(f"✅ 模型輸出摘要（前 200 字）：\n{summary[:200]}{'...' if len(summary) > 200 else ''}")
        return summary

    except Exception as e:
        print(f"⚠️ 壓縮時發生例外錯誤：{str(e)}")
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
    print(f"📝 使用者輸入：{message}")

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
    print(f"📤 發送給模型的 prompt（前 300 字）：\n{prompt[:300]}{'...' if len(prompt) > 300 else ''}")

    def try_model(model_name, timeout_sec):
        try:
            print(f"🧠 嘗試使用模型：{model_name}（timeout={timeout_sec}s）")
            result = subprocess.run(
                ["ollama", "run", model_name],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=timeout_sec
            )
            print(f"🔚 模型 {model_name} 回傳碼：{result.returncode}")
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8")
                print(f"⚠️ 模型 {model_name} 執行失敗：{stderr}")
                return None
            reply = result.stdout.decode("utf-8").strip()
            print(f"[分類判斷] 📥 回覆（前 200 字）：{reply[:200]}{'...' if len(reply) > 200 else ''}")
            return reply
        except subprocess.TimeoutExpired:
            print(f"⏰ 模型 {model_name} 超時（超過 {timeout_sec} 秒）")
            return None
        except Exception as e:
            print(f"❌ 模型 {model_name} 錯誤：{str(e)}")
            return None

    # 嘗試先用 phi4-mini，再 fallback 用 phi3:mini
    reply = try_model("phi4-mini", timeout_sec=120)
    if not reply:
        print("⚠️ phi4-mini 回覆失敗，改用 phi3:mini")
        reply = try_model("phi3:mini", timeout_sec=120)

    if not reply:
        print("⚠️ 無法取得分類結果，預設為 Semantic Query")
        return "Semantic Query"

    print("🧪 分析回覆內容進行分類...")
    if "Semantic Query" in reply:
        print("✅ 類別判斷：Semantic Query")
        return "Semantic Query"
    if "Statistical Analysis" in reply:
        print("✅ 類別判斷：Statistical Analysis")
        return "Statistical Analysis"
    if "Field Filter" in reply:
        print("✅ 類別判斷：Field Filter")
        return "Field Filter"
    if "Field Values" in reply:
        print("✅ 類別判斷：Field Values")
        return "Field Values"
    if "Temporal Trend" in reply:
        print("✅ 類別判斷：Temporal Trend")
        return "Temporal Trend"
    if "Solution Summary" in reply:
        print("✅ 類別判斷：Solution Summary")
        return "Solution Summary"

    # ❓ 落入 fallback
    print("⚠️ 回傳內容不在允許類型中，預設為 Semantic Query")
    return "Semantic Query"








# ----------- 統計分析查詢 -----------
def analyze_metadata_query(message):
    print("📊 執行統計分析...")
    print(f"📝 使用者輸入：{message}")

    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"📂 成功載入 metadata，總筆數：{len(metadata)}")
    except Exception as e:
        print("❌ metadata 載入錯誤")
        return f"⚠️ 無法載入 metadata：{str(e)}"

    system_prompt = (
        "You are helping analyze a structured knowledge base.\n"
        "From the user's question, choose ONE of the following fields to do statistical aggregation:\n"
        " - subcategory\n - configurationItem\n - roleComponent\n - location\n"
        "If the request is vague or unclear, respond with '__fallback__'.\n"
        "Only return one word: the field name or '__fallback__'.\n"
        "Do not return any explanation or code block. Just the field name."
    )
    prompt = f"{system_prompt}\n\nUser: {message}"
    print(f"📤 發送給模型的 prompt：\n{prompt}")

    try:
        print("🚀 呼叫模型 phi3:mini 分析欄位...")
        result = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=600
        )
        print(f"🔚 模型回傳碼：{result.returncode}")
        raw = result.stdout.decode("utf-8").strip()

        # ✅ 移除 markdown 格式包裹
        if raw.startswith("```"):
            print("⚠️ 偵測到 markdown 包裹，嘗試移除...")
            raw = raw.strip("`").strip()
            if "\n" in raw:
                raw = "\n".join(raw.split("\n")[1:-1])

        field = raw.strip().strip('"').strip("'").lower()  # ✅ 標準化字串
        print(f"[欄位判斷] GPT 回覆欄位：{field}")

        allowed_fields = {"subcategory", "configurationItem", "roleComponent", "location"}

        if field == "__fallback__":
            print("🔁 回傳 fallback，改為列出 subcategory 和 configurationItem 統計")
            return "\n\n".join([
                summarize_field("subcategory", metadata),
                summarize_field("configurationItem", metadata)
            ])

        if field not in allowed_fields:
            print("⚠️ 模型回傳不在允許欄位中")
            return f"⚠️ 無法判斷要統計的欄位（回覆為：{field}）"

        print(f"✅ 進行欄位 {field} 的統計")
        return summarize_field(field, metadata)

    except Exception as e:
        print(f"❌ 呼叫模型過程發生錯誤：{str(e)}")
        return f"⚠️ 呼叫模型分類欄位時出錯：{str(e)}"





# ----------- 統計欄位值 -----------
def summarize_field(field, metadata):
    print(f"📊 開始統計欄位：{field}")
    counts = {}
    for item in metadata:
        key = item.get(field, "未標註")
        counts[key] = counts.get(key, 0) + 1

    print(f"📈 統計完成，共有 {len(counts)} 種不同值")

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    for i, (k, v) in enumerate(sorted_counts[:5]):
        print(f"  🔢 Top{i+1}: {k} - {v} 筆")

    result_lines = [f"{i+1}. {k}：{v} 筆" for i, (k, v) in enumerate(sorted_counts[:5])]
    return f"📊 統計結果（依 {field}）：\n" + "\n".join(result_lines)


# ----------- 欄位值清單查詢 -----------
def list_field_values(message):
    print("🔍 啟動欄位值列舉任務...")
    print(f"📝 使用者提問：{message}")

    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"📂 成功載入 metadata，筆數：{len(metadata)}")
    except Exception as e:
        print("❌ metadata 載入失敗")
        return f"⚠️ Failed to load metadata: {str(e)}"

    system_prompt = (
        "You are a parser. The user is asking about what values are available in a certain field.\n"
        "Please extract which field they want to list.\n"
        "Return the field name only. Must be one of: configurationItem, subcategory, roleComponent, location"
    )
    prompt = f"{system_prompt}\n\nUser: {message}"
    print(f"📤 發送給模型的 prompt：\n{prompt}")

    try:
        print("🧠 使用模型 phi3:mini 判斷欄位名稱...")
        result = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=600
        )

        raw_reply = result.stdout.decode("utf-8").strip()
        print(f"[回應] GPT 回傳：{raw_reply}")

        field = raw_reply.strip()
        if field not in ["configurationItem", "subcategory", "roleComponent", "location"]:
            print("⚠️ 模型回傳無效欄位")
            return f"⚠️ Invalid field: {field}"

        print(f"✅ 欄位判定成功：{field}")
        values = set()
        for item in metadata:
            value = item.get(field)
            if value:
                values.add(value)

        print(f"📊 擷取欄位值完成，共 {len(values)} 種不同值")
        sorted_vals = sorted(values)
        for i, v in enumerate(sorted_vals[:5]):
            print(f"  - Top {i+1}: {v}")

        lines = [f"- {v}" for v in sorted_vals[:20]]
        return f"📋 Values in '{field}' field:\n" + "\n".join(lines)

    except Exception as e:
        print(f"❌ 呼叫模型或解析錯誤：{str(e)}")
        return f"⚠️ Failed to process: {str(e)}"

    


# ----------- 欄位查詢分析 -----------

def analyze_field_query(message):
    print("🔍 啟動多欄位條件查詢分析...")
    print(f"📝 使用者輸入：{message}")

    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"📂 成功載入 metadata，總筆數：{len(metadata)}")
    except Exception as e:
        print(f"❌ metadata 載入失敗：{str(e)}")
        return f"⚠️ Failed to load metadata: {str(e)}"

    # ✅ 取得合法值清單（限制模型輸出）
    allowed_fields = ["configurationItem", "subcategory", "roleComponent", "location"]
    valid_values = {field: sorted(set(str(item.get(field, "")).strip()) for item in metadata) for field in allowed_fields}
    value_hint = "\n".join([f"{field}: {valid_values[field]}" for field in allowed_fields])

    system_prompt = (
        "You are a parser. Extract all field=value conditions from the user's message for filtering.\n"
        "Only include fields: configurationItem, subcategory, roleComponent, location.\n"
        "Use only values from these lists:\n"
        f"{value_hint}\n"
        "Return ONLY a raw JSON array like:\n"
        '[{"field": "subcategory", "value": "Login/Access"}, {"field": "location", "value": "Taipei"}]\n'
        "Do not include any explanation or markdown formatting.\n"
        "The entire output must be a compact JSON array and must not exceed 500 characters in total."
    )

    prompt = f"{system_prompt}\n\nUser: {message}"
    print(f"📤 發送給模型的 prompt：\n{prompt}")

    try:
        print("🧠 呼叫模型 phi3:mini 判斷過濾欄位條件...")
        result = subprocess.run(
            ["ollama", "run", "phi3:mini"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=600
        )
        raw = result.stdout.decode("utf-8").strip()
        print("[🔍 多欄位查詢原始回覆]", raw)

        # ✅ 去除 markdown 包裹與標籤干擾
        if "```" in raw:
            print("⚠️ 偵測到 markdown 格式，正在清理...")
            raw = raw.split("```")[1].strip()
        if raw.startswith("json"):
            raw = raw[len("json"):].strip()

        print("📥 清理後的 JSON 字串：", raw)

        match = re.search(r'\[\s*{.*?}\s*\]', raw, re.DOTALL)
        if not match:
            return "⚠️ Failed to extract valid JSON array from model output."
        json_part = match.group(0)
        parsed_conditions = json.loads(json_part)
        print(f"✅ 成功解析為 JSON 陣列，共 {len(parsed_conditions)} 筆條件")

        if not isinstance(parsed_conditions, list):
            print("❌ 解析結果非 list 格式")
            return "⚠️ Invalid parsed result format (not a list)."

        filters = [(c["field"], c["value"]) for c in parsed_conditions if c.get("field") in allowed_fields]

        print("🔎 過濾後的有效條件：")
        for f, v in filters:
            print(f"  • {f} = {v}")

        if not filters:
            print("⚠️ 沒有擷取到有效條件")
            return "⚠️ No valid filters extracted from the query."

        # 篩選資料（符合所有條件，大小寫不敏感，空白容錯）
        def match_all(item):
            for field, value in filters:
                actual = str(item.get(field, "")).strip().lower()
                expected = str(value).strip().lower()
                if expected not in actual:  # ✅ 改為模糊比對
                    return False
            return True

        matches = [item for item in metadata if match_all(item)]

        print(f"📊 符合條件的結果筆數：{len(matches)}")

        if not matches:
            print("📭 查無結果")
            return f"🔍 No results found for: " + " AND ".join([f"{f}={v}" for f, v in filters])

        lines = [f"- {item.get('text', '')[:500]}" for item in matches[:5]]
        return (
            f"🔎 Top matches for:\n" +
            "\n".join([f"• {f} = {v}" for f, v in filters]) +
            "\n\n" + "\n".join(lines)
        )

    except Exception as e:
        print(f"❌ 呼叫模型或解析過程出錯：{str(e)}")
        return f"⚠️ Failed to parse or search: {str(e)}"




# ----------- 時間趨勢分析 -----------
def analyze_temporal_trend(message):
    print("📈 啟動時間趨勢分析...")
    print(f"📝 使用者輸入：{message}")

    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        print("📦 第一筆資料：", metadata[0])
        print("🔍 是否有 analysisTime 欄位：", "analysisTime" in metadata[0])
        print(f"📂 成功載入 metadata，總筆數：{len(metadata)}")
    except Exception as e:
        print(f"❌ metadata 載入失敗：{str(e)}")
        return f"⚠️ 無法載入資料：{str(e)}"

    if not metadata:
        print("⚠️ metadata 為空")
        return "⚠️ 無資料可分析。"

    if "analysisTime" not in metadata[0]:
        print("⚠️ analysisTime 欄位不存在")
        return "⚠️ 缺少 analysisTime 欄位，無法分析趨勢。"

    print("🧪 開始轉換為 DataFrame 並處理時間欄位...")
    df = pd.DataFrame(metadata)
    print(f"📊 DataFrame 欄位：{list(df.columns)}")

    df["analysisTime"] = pd.to_datetime(df["analysisTime"], errors="coerce")
    initial_len = len(df)
    df = df.dropna(subset=["analysisTime"])
    print(f"🧹 處理無效時間後，剩餘筆數：{len(df)} / 原始 {initial_len}")

    print("📆 建立月份欄位並計算每月數量...")
    df["month"] = df["analysisTime"].dt.to_period("M")
    trend = df.groupby("month").size()
    print("📊 每月統計結果：")
    for month, count in trend.items():
        print(f"  • {month}: {count} 筆")

    # ✅ 用文字敘述每月趨勢
    summary_lines = ["📊 每月案件趨勢："]
    for month, count in trend.items():
        summary_lines.append(f"- {month.strftime('%Y-%m')}: {count} 筆")
    print("✅ 已轉換為純文字描述")

    return "\n".join(summary_lines)




# ----------- 儲存查詢上下文 -----------
def save_query_context(chat_id, query, result_type, filter_info=None, result_summary=None):
    filepath = f"chat_history/{chat_id}.json"
    print(f"📁 嘗試讀取對話記錄檔：{filepath}")

    # 嘗試讀取對話歷史
    try:
        with open(filepath, encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list):
            print("⚠️ 記錄格式錯誤（非 list），重新初始化歷史")
            history = []
        else:
            print(f"📖 成功讀取歷史記錄，現有筆數：{len(history)}")
    except Exception as e:
        print(f"⚠️ 無法讀取歷史，初始化為空：{e}")
        history = []

    # 準備 context 內容
    context = {
        "type": result_type,
        "query": query,
        "filters": filter_info,
        "summary": result_summary
    }
    print(f"🧠 準備儲存的 context：{context}")

    # 若無歷史，建立佔位對話
    if not history:
        print("📌 尚無對話歷史，自動新增一則佔位訊息並附加 context。")
        history.append({
            "role": "user",
            "content": query,
            "context": context
        })
    else:
        print("🔁 已有歷史，將 context 寫入最後一則對話...")
        history[-1]["context"] = context

    # 顯示將要寫入的完整歷史
    print("📦 即將儲存的完整歷史內容預覽（最後 1 筆）：")
    print(json.dumps(history[-1], ensure_ascii=False, indent=2))

    # 儲存回 JSON 檔案
    try:
        os.makedirs("chat_history", exist_ok=True)  # 確保資料夾存在
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"✅ 已成功寫入檔案：{filepath}")
    except Exception as e:
        print(f"❌ 儲存記憶失敗：{e}")







# ----------- 延伸查詢處理 -----------
def is_follow_up_query(message: str) -> bool:
    print("🧠 判斷是否為追問查詢...")
    print(f"📝 輸入訊息：{message}")

    keywords = ["previous", "last query", "those", "add filter", "now show", "continue", "follow up"]
    lowered = message.lower()
    print(f"🔍 轉為小寫後訊息：{lowered}")

    for kw in keywords:
        if kw in lowered:
            print(f"✅ 命中關鍵字：'{kw}' → 判定為追問查詢")
            return True

    print("❌ 無關鍵字命中 → 非追問查詢")
    return False



# 處理追問查詢
def handle_follow_up(chat_id, message):
    filepath = f"chat_history/{chat_id}.json"
    print(f"📂 嘗試讀取歷史記錄：{filepath}")

    try:
        with open(filepath, encoding="utf-8") as f:
            history = json.load(f)
        print(f"📖 歷史筆數：{len(history)}")
    except Exception as e:
        print(f"❌ 歷史讀取失敗：{e}")
        return "⚠️ 無法讀取先前對話記錄，請確認 chat_id 是否正確。"

    if not history or "context" not in history[-1]:
        print("⚠️ 最後一筆歷史無 context 欄位")
        return "⚠️ 查無先前查詢條件，請重新描述您的需求。"

    context = history[-1]["context"]
    result_type = context.get("type")
    print(f"🧠 上次查詢類型為：{result_type}")

    if result_type == "Field Filter":
        original = context.get("filters", {})
        print(f"🔎 原始過濾條件：{original}")

        new_filter_prompt = (
            "You are a filter parser. Based on this message, extract an additional field and value to add as a filter.\n"
            "Return JSON like: {\"field\": \"subcategory\", \"value\": \"Crash\"}\n"
            "The entire response must be in compact JSON format and must not exceed 500 characters."
        )

        prompt = f"{new_filter_prompt}\n\nUser: {message}"
        print("📤 發送給模型的延伸過濾 prompt：")
        print(prompt)

        try:
            print("🧠 呼叫模型 phi3:mini 解析新增條件...")
            result = subprocess.run(
                ["ollama", "run", "phi3:mini"],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=600
            )
            raw_reply = result.stdout.decode("utf-8").strip()
            print(f"📥 GPT 回覆：{raw_reply}")

            new_filter = json.loads(raw_reply)
            field = new_filter.get("field")
            value = new_filter.get("value")
            print(f"✅ 擷取到欄位：{field}，值：{value}")

            if field not in ["configurationItem", "subcategory", "roleComponent", "location"]:
                print("⚠️ 欄位不在允許清單中")
                return "⚠️ 無效的欄位"

            filters = [original] + [new_filter]
            print(f"🔗 合併過濾條件：{filters}")

            with open("kb_metadata.json", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"📦 載入 metadata，總筆數：{len(metadata)}")

            matches = metadata
            for f in filters:
                matches = [m for m in matches if m.get(f["field"]) == f["value"]]
            print(f"📊 符合條件筆數：{len(matches)}")

            lines = [f"- {item.get('text', '')[:500]}" for item in matches[:5]]
            return f"🔎 延伸查詢結果（共 {len(matches)} 筆）：\n" + "\n".join(lines)

        except Exception as e:
            print(f"❌ 延伸查詢錯誤：{str(e)}")
            return f"⚠️ 延伸查詢失敗：{str(e)}"

    print("⚠️ 目前僅支援 Field Filter 類型的追問")
    return "⚠️ 目前只支援欄位篩選的延伸查詢。"

# ----------- 解法統整 -----------
def summarize_solutions(message):
    print("🛠️ 啟動相似案例處理方案摘要流程...")
    print(f"📝 使用者輸入：{message}")

    try:
        with open("kb_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"📂 成功載入 metadata，筆數：{len(metadata)}")
    except Exception as e:
        print(f"❌ metadata 載入失敗：{str(e)}")
        return f"⚠️ Failed to load metadata: {str(e)}"

    print("🔍 開始語意比對相關案例...")
    related_cases = search_knowledge_base(message, top_k=5)
    print(f"🧠 取得相似片段數量：{len(related_cases)}")
    for i, r in enumerate(related_cases, 1):
        print(f"  {i}. {r[:60]}{'...' if len(r) > 60 else ''}")

    if not related_cases:
        print("⚠️ 無相關案例")
        return "⚠️ No similar cases found to extract solutions."

    print("📦 擷取相關案例中的解決方案...")
    solutions = []
    for item in metadata:
        text = item.get("text", "")
        if any(snippet in text for snippet in related_cases):
            solution = item.get("solution", "")
            if solution:
                solutions.append(solution)

    print(f"✅ 擷取到 solution 數量：{len(solutions)}")
    if not solutions:
        print("⚠️ 無對應解決方案欄位")
        return "⚠️ No resolution data found for related cases."

    print("📝 組裝 prompt 進行 GPT 統整...")
    prompt = "Please summarize the following resolution steps into a brief, clear list:\n\n"
    prompt += "\n---\n".join(solutions[:10])
    prompt += "\n\nSummary:"

    print("📤 發送 prompt 給模型（前 300 字）：")
    print(prompt[:300] + ("..." if len(prompt) > 300 else ""))

    try:
        result = subprocess.run(
            ["ollama", "run", "phi4"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=600
        )
        output = result.stdout.decode("utf-8").strip()
        print("✅ 模型成功回應（前 200 字）：")
        print(output[:200] + ("..." if len(output) > 200 else ""))
        return output

    except Exception as e:
        print(f"❌ 模型摘要失敗：{str(e)}")
        return f"⚠️ Failed to summarize solutions: {str(e)}"






# ----------- GPT 主函式 -----------
def run_offline_gpt(message, model="mistral", history=[], chat_id=None):
    print("🟢 啟動 GPT 回答流程...")
    print(f"📝 使用者輸入：{message}")
    print(f"🧠 使用模型：{model} / chat_id: {chat_id}")

    query_type = classify_query_type(message)
    print(f"🔍 判斷結果：{query_type}")

    if is_follow_up_query(message) and chat_id:
        print("🔁 偵測為追問查詢，轉交 handle_follow_up 處理...")
        return handle_follow_up(chat_id, message)

    if query_type == "Statistical Analysis":
        print("📊 類型為統計分析，開始處理...")
        reply = analyze_metadata_query(message)
        print("📦 統計分析完成，摘要前 200 字：", reply[:200])
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply

    if query_type == "Field Filter":
        print("🔄 類型為欄位過濾，開始進行過濾...")
        reply = analyze_field_query(message)
        print("📦 過濾查詢完成，前 200 字：", reply[:200])

        filters = []
        for line in reply.splitlines():
            if line.strip().startswith("• "):
                try:
                    field_part = line.replace("•", "").strip()
                    field, value = field_part.split("=", 1)
                    filters.append({"field": field.strip(), "value": value.strip()})
                except Exception as e:
                    print(f"⚠️ 無法解析條件行：{line}，錯誤：{e}")
                    continue

        print("🧾 擷取到的 filters：", filters)
        save_query_context(chat_id, message, query_type, filter_info=filters if filters else None, result_summary=reply[:200])
        return reply

    if query_type == "Field Values":
        print("📋 類型為欄位值清單，開始列出欄位值...")
        reply = list_field_values(message)
        print("📦 欄位值查詢完成，前 200 字：", reply[:200])
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply

    if query_type == "Temporal Trend":
        print("📈 類型為時間趨勢查詢，開始繪製圖表...")
        reply = analyze_temporal_trend(message)
        print("📦 趨勢圖完成（HTML 片段）")
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply

    if query_type == "Solution Summary":
        print("🛠 類型為解法統整，開始彙整處理方式...")
        reply = summarize_solutions(message)
        print("📦 解法統整完成，前 200 字：", reply[:200])
        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply

    # 預設為 Semantic Query
    print("🔄 類型為語意查詢，開始檢索知識庫...")
    retrieved = search_knowledge_base(message, top_k=3)

    if retrieved:
        print(f"[RAG] ✅ 找到 {len(retrieved)} 筆相似資料：")
        for i, chunk in enumerate(retrieved, 1):
            preview = chunk.replace('\n', ' ')[:100]
            print(f"    {i}. {preview}...")
    else:
        print("[RAG] ⚠️ 未找到相似資料")

    print(f"[🔧 壓縮用模型] 使用模型：phi4-mini")
    print(f"[🎯 回答用模型] 使用模型：{model}")

    kb_context = summarize_retrieved_kb(retrieved, model="phi4-mini")
    print("📚 知識庫摘要完成")

    # 組合對話歷史
    context = ""
    if not isinstance(history, list):
        print("⚠️ 對話歷史格式錯誤，初始化為空 list")
        history = []
    for turn in history[-5:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        context += f"{role}: {turn['content']}\n"

    prompt = f"{kb_context}\n\n{context}User: {message}\nAssistant:"
    print("\n[Prompt Preview] 🧾 發送給模型的 Prompt 前 300 字：")
    print(prompt[:300] + ("..." if len(prompt) > 300 else ""))

    try:
        print("🚀 發送 prompt 給模型中...")
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=600
        )

        if result.returncode != 0:
            err = result.stderr.decode('utf-8')
            print("❌ 模型錯誤：", err)
            return f"⚠️ Ollama 錯誤：{err}"

        reply = result.stdout.decode("utf-8").strip()
        print("📥 模型回覆（前 300 字）：")
        print(reply[:300] + ("..." if len(reply) > 300 else ""))

        save_query_context(chat_id, message, query_type, result_summary=reply[:200])
        return reply if reply else "⚠️ 沒有收到模型回應。"

    except Exception as e:
        print(f"❌ 呼叫模型失敗：{str(e)}")
        return f"⚠️ 呼叫模型時發生錯誤：{str(e)}"
