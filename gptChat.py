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
import sqlite3

DB_PATH = "resultDB.db"  # 你在 build_kb.py 裡設定的 DB 名稱



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
        print("⚠️ 無資料可摘要（retrieved 為空）")
        return ""

    print("🧠 正在進行分段摘要處理（retrieved KB）...")
    print(f"📦 輸入筆數：{len(retrieved)}")

    model_token_limits = {
        "phi4-mini": 4096,
        "phi3:mini": 4096,
        "mistral": 8192,
        "deepseek-coder:latest": 16384,
        "deepseek-coder-v2": 16384,
    }
    token_limit = model_token_limits.get(model, 4096)
    prompt_reserve = 500
    available_tokens = token_limit - prompt_reserve

    def estimate_token(text):
        return int(len(text) / 4)

    # 分段
    groups = []
    group = []
    token_sum = 0
    for text in retrieved:
        tokens = estimate_token(text)
        if token_sum + tokens > available_tokens and group:
            groups.append(group)
            group = [text]
            token_sum = tokens
        else:
            group.append(text)
            token_sum += tokens
    if group:
        groups.append(group)

    # 對每段進行摘要
    chunk_summaries = []
    for i, group in enumerate(groups, 1):
        prompt = "Please summarize the key points and handling methods based on the following knowledge entries (respond in English):\n\n"
        for j, txt in enumerate(group, 1):
            prompt += f"{j}. {txt.strip()}\n"
        prompt += "\nPlease provide a single summary paragraph:"

        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=600
            )
            if result.returncode == 0:
                reply = result.stdout.decode("utf-8").strip()
                chunk_summaries.append(reply)
                print(f"✅ 摘要完成（第 {i} 組）")
            else:
                print(f"⚠️ 第 {i} 組摘要失敗，嘗試使用 fallback 模型 phi3:mini...")
                fallback_result = subprocess.run(
                    ["ollama", "run", "phi3:mini"],
                    input=prompt.encode("utf-8"),
                    capture_output=True,
                    timeout=600
                )
                if fallback_result.returncode == 0:
                    reply = fallback_result.stdout.decode("utf-8").strip()
                    chunk_summaries.append(reply)
                    print(f"✅ Fallback 成功（第 {i} 組）")
                else:
                    chunk_summaries.append("❌ 本段摘要失敗")
        except Exception as e:
            print(f"❌ 第 {i} 段呼叫模型失敗：{e}")
            chunk_summaries.append("❌ 摘要失敗")

    # 若只剩一組摘要，直接回傳
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # 再次檢查是否會超出 token 限制
    def recursive_merge(summaries):
        available_tokens = token_limit - prompt_reserve
        merged_groups = []
        group = []
        token_count = 0

        for s in summaries:
            t = estimate_token(s)
            if token_count + t > available_tokens and group:
                merged_groups.append(group)
                group = [s]
                token_count = t
            else:
                group.append(s)
                token_count += t
        if group:
            merged_groups.append(group)

        results = []
        for i, group in enumerate(merged_groups, 1):
            merge_prompt = "Based on the following summaries, please synthesize the main insights:\n\n"
            for j, s in enumerate(group, 1):
                merge_prompt += f"（第 {j} 段摘要）{s}\n\n"
            merge_prompt += "Please provide an overall concluding observation:"
            try:
                result = subprocess.run(
                    ["ollama", "run", model],
                    input=merge_prompt.encode("utf-8"),
                    capture_output=True,
                    timeout=600
                )
                if result.returncode == 0:
                    results.append(result.stdout.decode("utf-8").strip())
                else:
                    print(f"⚠️ 合併失敗（第 {i} 組），fallback 使用 phi3:mini")
                    fallback = subprocess.run(
                        ["ollama", "run", "phi3:mini"],
                        input=merge_prompt.encode("utf-8"),
                        capture_output=True,
                        timeout=600
                    )
                    if fallback.returncode == 0:
                        results.append(fallback.stdout.decode("utf-8").strip())
                    else:
                        results.append("❌ 合併失敗")
            except Exception as e:
                print(f"❌ 合併摘要時錯誤：{e}")
                results.append("❌ 合併摘要錯誤")

        return results[0] if len(results) == 1 else recursive_merge(results)

    return recursive_merge(chunk_summaries)




def determine_top_k_with_llm(user_input, fallback=3, model="phi4-mini", min_top_k=1, max_top_k=10):
    print("🧠 使用 LLM 預測合適的 top_k 數量...")

    prompt = (
        "You are a knowledge retrieval assistant. Based on the user's question, decide how many similar cases (top_k) should be retrieved from the knowledge base.\n\n"
        "Guidelines:\n"
        "- If the question is **very specific** (mentions error codes, clear symptoms, or keywords), return a **small** top_k (1–3).\n"
        "- If the question is **vague or general** (like 'why is it slow?' or 'something went wrong'), return a **larger** top_k (5–10).\n"
        "- If the user asks for a **summary, report, or trend**, use a **larger** top_k (8–10).\n"
        "- Only reply with a **single integer** between 1 and 10. Do not add explanation.\n\n"
        f"User question: {user_input}\n\nAnswer:"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=60
        )

        if result.returncode == 0:
            reply = result.stdout.decode("utf-8").strip()
            print(f"📥 LLM 回覆的 top_k 數值為：{reply}")
            try:
                top_k = int(reply)
                # Enforce bounds
                top_k = max(min_top_k, min(top_k, max_top_k))
                return top_k
            except:
                pass
        print("⚠️ 無法解析 top_k，改用 fallback")
        return fallback

    except Exception as e:
        print(f"❌ 呼叫 LLM 判斷 top_k 失敗：{e}")
        return fallback





# ----------- 知識庫檢索(語意比對類別) -----------
def search_knowledge_base(query, top_k=None):
    print("🔍 執行語意查詢...")

    if kb_model is None or kb_index is None or kb_texts is None:
        print("❌ 知識庫尚未載入")
        return []

    # 如果沒有指定 top_k，就自動判斷
    if top_k is None:
        top_k = determine_top_k_with_llm(query)  # 呼叫 LLM 決定合適的 top_k
        print(f"🤖 動態決定 top_k = {top_k}")

    query_vec = kb_model.encode([query])
    D, I = kb_index.search(np.array(query_vec), top_k)
    print(f"[RAG] 🔍 查詢內容：{query}")
    print(f"[RAG] 🧠 取出知識庫資料：{[kb_texts[i][:50] for i in I[0]]}")
    return [kb_texts[i] for i in I[0]]






# ----------- 提問類型分類 -----------
def classify_query_type(message):
    print("🤖 判斷提問類型...")
    print(f"📝 使用者輸入：{message}")

    system_prompt = (
        "You are a classification assistant. Based on the user's question, determine whether it belongs to one of the following types:\n"
        "1. Semantic Query (user wants similar past cases or issue solving)\n"
        "2. Structured SQL (user wants structured data, including counts, filters, field values, temporal trends, or solution summaries)\n"
        "Please respond with one of: 'Semantic Query' or 'Structured SQL'."

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
    if "Structured SQL" in reply:
        print("✅ 類別判斷：Structured SQL")
        return "Structured SQL"

    # ❓ 落入 fallback
    print("⚠️ 回傳內容不在允許類型中，預設為 Semantic Query")
    return "Semantic Query"








# ----------- 統計分析查詢 -----------
# def analyze_metadata_query(message):
#     print("📊 執行統計分析...")
#     print(f"📝 使用者輸入：{message}")

#     try:
#         with open("kb_metadata.json", encoding="utf-8") as f:
#             metadata = json.load(f)
#         print(f"📂 成功載入 metadata，總筆數：{len(metadata)}")
#     except Exception as e:
#         print("❌ metadata 載入錯誤")
#         return f"⚠️ 無法載入 metadata：{str(e)}"

#     system_prompt = (
#         "You are helping analyze a structured knowledge base.\n"
#         "From the user's question, choose ONE of the following fields to do statistical aggregation:\n"
#         " - subcategory\n - configurationItem\n - roleComponent\n - location\n"
#         "If the request is vague or unclear, respond with '__fallback__'.\n"
#         "Only return one word: the field name or '__fallback__'.\n"
#         "Do not return any explanation or code block. Just the field name."
#     )
#     prompt = f"{system_prompt}\n\nUser: {message}"
#     print(f"📤 發送給模型的 prompt：\n{prompt}")

#     try:
#         print("🚀 呼叫模型 phi3:mini 分析欄位...")
#         result = subprocess.run(
#             ["ollama", "run", "phi3:mini"],
#             input=prompt.encode("utf-8"),
#             capture_output=True,
#             timeout=600
#         )
#         print(f"🔚 模型回傳碼：{result.returncode}")
#         raw = result.stdout.decode("utf-8").strip()

#         # ✅ 移除 markdown 格式包裹
#         if raw.startswith("```"):
#             print("⚠️ 偵測到 markdown 包裹，嘗試移除...")
#             raw = raw.strip("`").strip()
#             if "\n" in raw:
#                 raw = "\n".join(raw.split("\n")[1:-1])

#         field = raw.strip().strip('"').strip("'").lower()  # ✅ 標準化字串
#         print(f"[欄位判斷] GPT 回覆欄位：{field}")

#         allowed_fields = {"subcategory", "configurationItem", "roleComponent", "location"}

#         if field == "__fallback__":
#             print("🔁 回傳 fallback，改為列出 subcategory 和 configurationItem 統計")
#             return "\n\n".join([
#                 summarize_field("subcategory", metadata),
#                 summarize_field("configurationItem", metadata)
#             ])

#         if field not in allowed_fields:
#             print("⚠️ 模型回傳不在允許欄位中")
#             return f"⚠️ 無法判斷要統計的欄位（回覆為：{field}）"

#         print(f"✅ 進行欄位 {field} 的統計")
#         return summarize_field(field, metadata)

#     except Exception as e:
#         print(f"❌ 呼叫模型過程發生錯誤：{str(e)}")
#         return f"⚠️ 呼叫模型分類欄位時出錯：{str(e)}"





# ----------- 統計欄位值 -----------
# def summarize_field(field, metadata):
#     print(f"📊 開始統計欄位：{field}")
#     counts = {}
#     for item in metadata:
#         key = item.get(field, "未標註")
#         counts[key] = counts.get(key, 0) + 1

#     print(f"📈 統計完成，共有 {len(counts)} 種不同值")

#     sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
#     for i, (k, v) in enumerate(sorted_counts[:5]):
#         print(f"  🔢 Top{i+1}: {k} - {v} 筆")

#     result_lines = [f"{i+1}. {k}：{v} 筆" for i, (k, v) in enumerate(sorted_counts[:5])]
#     return f"📊 統計結果（依 {field}）：\n" + "\n".join(result_lines)


# ----------- 欄位值清單查詢 -----------
# def list_field_values(message):
#     print("🔍 啟動欄位值列舉任務...")
#     print(f"📝 使用者提問：{message}")

#     try:
#         with open("kb_metadata.json", encoding="utf-8") as f:
#             metadata = json.load(f)
#         print(f"📂 成功載入 metadata，筆數：{len(metadata)}")
#     except Exception as e:
#         print("❌ metadata 載入失敗")
#         return f"⚠️ Failed to load metadata: {str(e)}"

#     system_prompt = (
#         "You are a parser. The user is asking about what values are available in a certain field.\n"
#         "Please extract which field they want to list.\n"
#         "Return the field name only. Must be one of: configurationItem, subcategory, roleComponent, location"
#     )
#     prompt = f"{system_prompt}\n\nUser: {message}"
#     print(f"📤 發送給模型的 prompt：\n{prompt}")

#     try:
#         print("🧠 使用模型 phi3:mini 判斷欄位名稱...")
#         result = subprocess.run(
#             ["ollama", "run", "phi3:mini"],
#             input=prompt.encode("utf-8"),
#             capture_output=True,
#             timeout=600
#         )

#         raw_reply = result.stdout.decode("utf-8").strip()
#         print(f"[回應] GPT 回傳：{raw_reply}")

#         field = raw_reply.strip()
#         if field not in ["configurationItem", "subcategory", "roleComponent", "location"]:
#             print("⚠️ 模型回傳無效欄位")
#             return f"⚠️ Invalid field: {field}"

#         print(f"✅ 欄位判定成功：{field}")
#         values = set()
#         for item in metadata:
#             value = item.get(field)
#             if value:
#                 values.add(value)

#         print(f"📊 擷取欄位值完成，共 {len(values)} 種不同值")
#         sorted_vals = sorted(values)
#         for i, v in enumerate(sorted_vals[:5]):
#             print(f"  - Top {i+1}: {v}")

#         lines = [f"- {v}" for v in sorted_vals[:20]]
#         return f"📋 Values in '{field}' field:\n" + "\n".join(lines)

#     except Exception as e:
#         print(f"❌ 呼叫模型或解析錯誤：{str(e)}")
#         return f"⚠️ Failed to process: {str(e)}"

    


# ----------- 欄位查詢分析 -----------

# def analyze_field_query(message):
#     print("🔍 啟動多欄位條件查詢分析...")
#     print(f"📝 使用者輸入：{message}")

#     try:
#         with open("kb_metadata.json", encoding="utf-8") as f:
#             metadata = json.load(f)
#         print(f"📂 成功載入 metadata，總筆數：{len(metadata)}")
#     except Exception as e:
#         print(f"❌ metadata 載入失敗：{str(e)}")
#         return f"⚠️ Failed to load metadata: {str(e)}"

#     # ✅ 取得合法值清單（限制模型輸出）
#     allowed_fields = ["configurationItem", "subcategory", "roleComponent", "location"]
#     valid_values = {
#         field: sorted(set(str(item.get(field, "")).strip() for item in metadata if item.get(field)))
#         for field in allowed_fields
#     }
#     value_hint = "\n".join([f"{field}: {valid_values[field]}" for field in allowed_fields])

#     system_prompt = (
#         "You are a parser. Extract all field=value conditions from the user's message for filtering.\n"
#         "Only include fields: configurationItem, subcategory, roleComponent, location.\n"
#         "Use only values from these lists:\n"
#         f"{value_hint}\n"
#         "Return ONLY a raw JSON array like:\n"
#         '[{"field": "subcategory", "value": "Login/Access"}, {"field": "location", "value": "Taipei"}]\n'
#         "Do not include any explanation or markdown formatting.\n"
#         "The entire output must be a compact JSON array and must not exceed 500 characters in total."
#     )

#     prompt = f"{system_prompt}\n\nUser: {message}"
#     print(f"📤 發送給模型的 prompt：\n{prompt}")

#     try:
#         print("🧠 呼叫模型 phi3:mini 判斷過濾欄位條件...")
#         result = subprocess.run(
#             ["ollama", "run", "phi3:mini"],
#             input=prompt.encode("utf-8"),
#             capture_output=True,
#             timeout=600
#         )
#         raw = result.stdout.decode("utf-8").strip()
#         print("[🔍 多欄位查詢原始回覆]", raw)

#     # ✅ 去除 markdown 包裹與標籤干擾
#         if "```" in raw:
#             print("⚠️ 偵測到 markdown 格式，正在清理...")
#             raw = raw.split("```")[1].strip()
#         if raw.startswith("json"):
#             raw = raw[len("json"):].strip()

#         print("📥 清理後的 JSON 字串：", raw)

#         # ✅ 嘗試從原始字串中擷取合法 JSON 陣列
#         match = re.search(r'\[\s*{.*?}\s*\]', raw, re.DOTALL)
#         if not match:
#             return "⚠️ Failed to extract valid JSON array from model output."
#         json_part = match.group(0)

#         try:
#             parsed_conditions = json.loads(json_part)
#             print(f"✅ 成功解析為 JSON 陣列，共 {len(parsed_conditions)} 筆條件")
#         except Exception as e:
#             print(f"❌ JSON 解析失敗：{e}")
#             return f"⚠️ JSON decode error: {str(e)}"


#         if not isinstance(parsed_conditions, list):
#             print("❌ 解析結果非 list 格式")
#             return "⚠️ Invalid parsed result format (not a list)."

#         filters = [(c["field"], c["value"]) for c in parsed_conditions if c.get("field") in allowed_fields]

#         print("🔎 過濾後的有效條件：")
#         for f, v in filters:
#             print(f"  • {f} = {v}")

#         if not filters:
#             print("⚠️ 沒有擷取到有效條件")
#             return "⚠️ No valid filters extracted from the query."

#         # 篩選資料（符合所有條件，大小寫不敏感，空白容錯）
#         def match_all(item):
#             for field, value in filters:
#                 actual = str(item.get(field, "")).strip().lower()
#                 expected = str(value).strip().lower()
#                 if expected not in actual:  # ✅ 改為模糊比對
#                     return False
#             return True

#         matches = [item for item in metadata if match_all(item)]

#         print(f"📊 符合條件的結果筆數：{len(matches)}")

#         if not matches:
#             print("📭 查無結果")
#             return f"🔍 No results found for: " + " AND ".join([f"{f}={v}" for f, v in filters])

#         lines = [f"- {item.get('text', '')[:500]}" for item in matches[:5]]
#         # 🔁 從 matches 中取出實際命中的原始值
#         actual_values = {field: set() for field, _ in filters}
#         for item in matches:
#             for field, _ in filters:
#                 val = item.get(field, "").strip()
#                 if val:
#                     actual_values[field].add(val)

#         summary_lines = [
#             f"• {field} = {', '.join(sorted(actual_values[field])) or 'N/A'}"
#             for field in actual_values
#         ]

#         return (
#             "🔎 Top matches for:\n" +
#             "\n".join(summary_lines) +
#             "\n\n" + "\n".join(lines)
#         )
    
#     except Exception as e:
#         print(f"❌ 呼叫模型或解析過程出錯：{str(e)}")
#         return f"⚠️ Failed to parse or search: {str(e)}"




# ----------- 時間趨勢分析 -----------
# def analyze_temporal_trend(message):
#     print("📈 啟動時間趨勢分析...")
#     print(f"📝 使用者輸入：{message}")

#     try:
#         with open("kb_metadata.json", encoding="utf-8") as f:
#             metadata = json.load(f)
#         print("📦 第一筆資料：", metadata[0])
#         print("🔍 是否有 analysisTime 欄位：", "analysisTime" in metadata[0])
#         print(f"📂 成功載入 metadata，總筆數：{len(metadata)}")
#     except Exception as e:
#         print(f"❌ metadata 載入失敗：{str(e)}")
#         return f"⚠️ 無法載入資料：{str(e)}"

#     if not metadata:
#         print("⚠️ metadata 為空")
#         return "⚠️ 無資料可分析。"

#     if "analysisTime" not in metadata[0]:
#         print("⚠️ analysisTime 欄位不存在")
#         return "⚠️ 缺少 analysisTime 欄位，無法分析趨勢。"

#     print("🧪 開始轉換為 DataFrame 並處理時間欄位...")
#     df = pd.DataFrame(metadata)
#     print(f"📊 DataFrame 欄位：{list(df.columns)}")

#     df["analysisTime"] = pd.to_datetime(df["analysisTime"], errors="coerce")
#     initial_len = len(df)
#     df = df.dropna(subset=["analysisTime"])
#     print(f"🧹 處理無效時間後，剩餘筆數：{len(df)} / 原始 {initial_len}")

#     print("📆 建立月份欄位並計算每月數量...")
#     df["month"] = df["analysisTime"].dt.to_period("M")
#     trend = df.groupby("month").size()
#     print("📊 每月統計結果：")
#     for month, count in trend.items():
#         print(f"  • {month}: {count} 筆")

#     # ✅ 用文字敘述每月趨勢
#     summary_lines = ["📊 每月案件趨勢："]
#     for month, count in trend.items():
#         summary_lines.append(f"- {month.strftime('%Y-%m')}: {count} 筆")
#     print("✅ 已轉換為純文字描述")

#     return "\n".join(summary_lines)




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
# def summarize_solutions(message):
#     print("🛠️ 啟動相似案例處理方案摘要流程...")
#     print(f"📝 使用者輸入：{message}")

#     try:
#         with open("kb_metadata.json", encoding="utf-8") as f:
#             metadata = json.load(f)
#         print(f"📂 成功載入 metadata，筆數：{len(metadata)}")
#     except Exception as e:
#         print(f"❌ metadata 載入失敗：{str(e)}")
#         return f"⚠️ Failed to load metadata: {str(e)}"

#     print("🔍 開始語意比對相關案例...")
#     related_cases = search_knowledge_base(message, top_k=5)
#     print(f"🧠 取得相似片段數量：{len(related_cases)}")
#     for i, r in enumerate(related_cases, 1):
#         print(f"  {i}. {r[:60]}{'...' if len(r) > 60 else ''}")

#     if not related_cases:
#         print("⚠️ 無相關案例")
#         return "⚠️ No similar cases found to extract solutions."

#     print("📦 擷取相關案例中的解決方案...")
#     solutions = []
#     for item in metadata:
#         text = item.get("text", "")
#         if any(snippet in text for snippet in related_cases):
#             solution = item.get("solution", "")
#             if solution:
#                 solutions.append(solution)

#     print(f"✅ 擷取到 solution 數量：{len(solutions)}")
#     if not solutions:
#         print("⚠️ 無對應解決方案欄位")
#         return "⚠️ No resolution data found for related cases."

#     print("📝 組裝 prompt 進行 GPT 統整...")
#     prompt = "Please summarize the following resolution steps into a brief, clear list:\n\n"
#     prompt += "\n---\n".join(solutions[:10])
#     prompt += "\n\nSummary:"

#     print("📤 發送 prompt 給模型（前 300 字）：")
#     print(prompt[:300] + ("..." if len(prompt) > 300 else ""))

#     try:
#         result = subprocess.run(
#             ["ollama", "run", "phi4"],
#             input=prompt.encode("utf-8"),
#             capture_output=True,
#             timeout=600
#         )
#         output = result.stdout.decode("utf-8").strip()
#         print("✅ 模型成功回應（前 200 字）：")
#         print(output[:200] + ("..." if len(output) > 200 else ""))
#         return output

#     except Exception as e:
#         print(f"❌ 模型摘要失敗：{str(e)}")
#         return f"⚠️ Failed to summarize solutions: {str(e)}"
    

def build_sql_prompt(user_question):
    schema_info = """
        You are an expert data analyst.

        You are working with the following SQLite table named 'metadata':

        Columns:
        - id (integer): internal ID
        - text (text): full case description, includes issue and solution
        - subcategory (text): issue type, such as 'Login', 'Teams', etc.
        - configurationItem (text): module or system component
        - roleComponent (text): affected user role or feature
        - location (text): site or region where issue occurred
        - analysisTime (text): ISO timestamp when the issue was recorded

        Please write an SQL query (SELECT ...) to answer the user's question.
        Return only the SQL query, no explanation or formatting.
        """

    prompt = f"{schema_info}\n\nUser question: {user_question}\nSQL:"
    return prompt

def generate_sql_with_llm(prompt, model="mistral"):
    try:
        print("🚀 呼叫模型產生 SQL 中...")
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=600
        )

        if result.returncode != 0:
            err = result.stderr.decode("utf-8")
            print("❌ 模型錯誤：", err)
            return None

        output = result.stdout.decode("utf-8").strip()
        print("📥 模型產出（前 200 字）：", output[:200])
        return output

    except Exception as e:
        print(f"❌ 呼叫 LLM 失敗：{str(e)}")
        return None



def extract_sql_code(text):
    print("🔍 嘗試從模型回應中萃取 SQL 指令...")

    # 嘗試抓 ```sql 區塊（修正 \\s -> \s）
    code_block_match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        sql_code = code_block_match.group(1).strip()
        print("✅ 偵測到 ```sql 區塊，成功抽取 SQL。")
        return sql_code

    # 否則抓第一段 SELECT 語句（也修正 \\s）
    select_match = re.search(r"(SELECT\s.+?;)", text, re.IGNORECASE | re.DOTALL)
    if select_match:
        sql_code = select_match.group(1).strip()
        print("✅ 成功抽取 SELECT 開頭的 SQL。")
        return sql_code

    # 最後 fallback：試著抓整段包含 FROM 的段落
    fallback_match = re.search(r"(SELECT.+FROM.+?)(\n|$)", text, re.IGNORECASE | re.DOTALL)
    if fallback_match:
        sql_code = fallback_match.group(1).strip()
        print("⚠️ 從 fallback 抽取 SQL 成功（但可能不完整）。")
        return sql_code

    print("⚠️ 無法抽取 SQL，原始輸出如下：")
    print(text[:300])
    return None




def run_sql(query):
    try:
        print("🔍 正在連線到 SQLite 資料庫...")
        conn = sqlite3.connect(DB_PATH)
        print("🔍 正在查詢 SQLite 資料庫...")
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"✅ 查詢成功，共 {len(df)} 筆結果。")
        return df
    except Exception as e:
        print("❌ 查詢失敗：", e)
        return None

# ---------- 人類摘要 ----------
def summarize_sql_result(df, max_rows=5):
    if df.empty:
        return "📭 No data found."

    preview_df = df.head(max_rows).copy()

    # 將文字欄位內的 \n 轉為實際換行
    for col in preview_df.columns:
        if preview_df[col].dtype == "object":
            preview_df[col] = preview_df[col].astype(str).str.replace("\\n", "\n").str.slice(0, 200)

    summary = f"📊 Query successful. Total {len(df)} records found.\n"
    summary += f"📋 Preview of first {min(max_rows, len(df))} records:\n"
    preview = preview_df.to_string(index=False)

    return summary + "```\n" + preview + "\n```"

def estimate_tokens_per_row(df):
    csv_text = df.to_csv(index=False)
    avg_len = len(csv_text) / len(df) if len(df) > 0 else 1
    # 粗略推估：1 token ≈ 4 個字元
    return int(avg_len / 4)


def calculate_dynamic_chunk_size(df, model_name="phi4-mini", prompt_reserve_tokens=500):
    model_token_limits = {
        "phi4-mini": 4096,
        "phi3:mini": 4096,
        "mistral": 8192,
        "orca2": 8192,
        "deepseek-coder:latest": 16384,
        "deepseek-coder-v2": 16384,
    }
    max_tokens = model_token_limits.get(model_name, 4096)
    tokens_per_row = estimate_tokens_per_row(df)
    available_tokens = max_tokens - prompt_reserve_tokens
    # 🛡️ 為了輸出品質，再減 10 筆
    chunk_size = max(5, int(available_tokens / tokens_per_row)-10)
    return min(chunk_size, len(df))

def estimate_token_count(text):
    return len(text) // 4




def split_and_merge_summaries(summaries, model, token_limit=4096, prompt_reserve=500):
    available_tokens = token_limit - prompt_reserve
    grouped = []
    group = []
    token_count = 0

    for s in summaries:
        s_tokens = estimate_token_count(s)
        if token_count + s_tokens > available_tokens:
            grouped.append(group)
            group = [s]
            token_count = s_tokens
        else:
            group.append(s)
            token_count += s_tokens
    if group:
        grouped.append(group)

    merged_chunks = []
    for i, group in enumerate(grouped, 1):
        merge_prompt = f"You are a data analyst. Please summarize the key points from the following group {i} of summaries:\n\n"

        for idx, s in enumerate(group, 1):
            merge_prompt += f"（摘要 {idx}）{s}\n\n"
        merge_prompt += "Please consolidate the main observations:"

        def run_with_model(m):
            try:
                result = subprocess.run(
                    ["ollama", "run", m],
                    input=merge_prompt.encode("utf-8"),
                    capture_output=True,
                    timeout=300
                )
                if result.returncode == 0:
                    return result.stdout.decode("utf-8").strip()
                else:
                    print(f"⚠️ 模型 {m} 回傳失敗")
                    return None
            except Exception as e:
                print(f"❌ 模型 {m} 發生錯誤：{e}")
                return None

        # 嘗試主模型
        reply = run_with_model(model)
        if not reply:
            print("🔁 使用 fallback 模型：phi3:mini")
            reply = run_with_model("phi3:mini")

        merged_chunks.append(reply if reply else "❌ 本段摘要失敗")

    if len(merged_chunks) == 1:
        return f"📊 GPT 整合摘要如下：\n{merged_chunks[0]}"
    else:
        return split_and_merge_summaries(merged_chunks, model, token_limit, prompt_reserve)








def summarize_sql_result_with_llm(df, model="phi4-mini"):
    if df.empty:
        return "📭 查無資料結果。"

    chunk_size = calculate_dynamic_chunk_size(df, model)
    print(f"📐 預估 chunk_size = {chunk_size} 筆（模型：{model}）")

    chunk_summaries = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        sample_csv = chunk.to_csv(index=False)
        prompt = f"You are a data analyst. The following is data chunk {i//chunk_size+1}. Please summarize its characteristics and trends:\n\n{sample_csv}\n\nSummary:"

        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=600
            )

            if result.returncode == 0:
                reply = result.stdout.decode("utf-8").strip()
                chunk_summaries.append(reply)
                print(f"✅ 第 {i//chunk_size+1} 段完成摘要")
            else:
                print(f"⚠️ 第 {i//chunk_size+1} 段摘要失敗，跳過")

        except Exception as e:
            print(f"❌ 第 {i//chunk_size+1} 段呼叫 LLM 失敗：{e}")

    if not chunk_summaries:
        return summarize_sql_result(df)
    
    # 開始整合
    print("🧠 開始整合所有段落摘要...")

    merge_prompt = "You are a data analyst. Based on the following multiple summaries, please provide an overall conclusion:\n\n"
    for idx, s in enumerate(chunk_summaries, 1):
        merge_prompt += f"（第 {idx} 段摘要）{s}\n\n"
    merge_prompt += "Please provide the key insights and observations:"

    def run_with_fallback(prompt, primary_model, fallback_model="phi3:mini"):
        try:
            result = subprocess.run(
                ["ollama", "run", primary_model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=600
            )
            if result.returncode == 0:
                return result.stdout.decode("utf-8").strip()
            else:
                print(f"⚠️ 主模型 {primary_model} 失敗，嘗試 fallback 模型 {fallback_model}")
        except Exception as e:
            print(f"❌ 主模型 {primary_model} 發生錯誤：{e}")

        # 嘗試 fallback 模型
        try:
            result = subprocess.run(
                ["ollama", "run", fallback_model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=600
            )
            if result.returncode == 0:
                return result.stdout.decode("utf-8").strip()
            else:
                print(f"⚠️ fallback 模型 {fallback_model} 也失敗")
        except Exception as e:
            print(f"❌ fallback 模型 {fallback_model} 發生錯誤：{e}")

        return None

    final_summary = run_with_fallback(merge_prompt, model)
    if final_summary:
        return f"📊 GPT 整合摘要如下：\n{final_summary}"
    else:
        print("⚠️ 合併摘要失敗，回傳各段摘要集合")
        return "\n\n".join(chunk_summaries)


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
    
    if query_type == "Structured SQL":
        print("🧾 類型為 SQL 結構化查詢，開始生成 SQL...")
        refined_prompt = build_sql_prompt(message)
        raw_sql = generate_sql_with_llm(refined_prompt)
        sql_code = extract_sql_code(raw_sql)

        if not sql_code:
            return "⚠️ 無法從 LLM 回覆中抽取有效的 SQL 指令。"

        df = run_sql(sql_code)
        if df is None or df.empty:
            return "📭 查無資料結果，請調整條件後再試。"

        summary = summarize_sql_result(df)
        summaryByLLM = summarize_sql_result_with_llm(df)

        combined_summary = (
            "📋 [系統摘要]\n" + summary.strip() +
            "\n\n🧠 [GPT 摘要]\n" + summaryByLLM.strip()
        )

        save_query_context(chat_id, message, query_type, result_summary=combined_summary[:500])

        return f"{summary}\n\n{summaryByLLM}"



    # if query_type == "Statistical Analysis":
    #     print("📊 類型為統計分析，開始處理...")
    #     reply = analyze_metadata_query(message)
    #     print("📦 統計分析完成，摘要前 200 字：", reply[:200])
    #     save_query_context(chat_id, message, query_type, result_summary=reply[:200])
    #     return reply

    # if query_type == "Field Filter":
    #     print("🔄 類型為欄位過濾，開始進行過濾...")
    #     reply = analyze_field_query(message)
    #     print("📦 過濾查詢完成，前 200 字：", reply[:200])
    #     filters = []
    #     for line in reply.splitlines():
    #         if line.strip().startswith("• "):
    #             try:
    #                 field_part = line.replace("•", "").strip()
    #                 field, value = field_part.split("=", 1)
    #                 filters.append({"field": field.strip(), "value": value.strip()})
    #             except Exception as e:
    #                 print(f"⚠️ 無法解析條件行：{line}，錯誤：{e}")
    #                 continue
    #     print("🧾 擷取到的 filters：", filters)
    #     save_query_context(chat_id, message, query_type, filter_info=filters if filters else None, result_summary=reply[:200])
    #     return reply

    # if query_type == "Field Values":
    #     print("📋 類型為欄位值清單，開始列出欄位值...")
    #     reply = list_field_values(message)
    #     print("📦 欄位值查詢完成，前 200 字：", reply[:200])
    #     save_query_context(chat_id, message, query_type, result_summary=reply[:200])
    #     return reply

    # if query_type == "Temporal Trend":
    #     print("📈 類型為時間趨勢查詢，開始繪製圖表...")
    #     reply = analyze_temporal_trend(message)
    #     print("📦 趨勢圖完成（HTML 片段）")
    #     save_query_context(chat_id, message, query_type, result_summary=reply[:200])
    #     return reply

    # if query_type == "Solution Summary":
    #     print("🛠 類型為解法統整，開始彙整處理方式...")
    #     reply = summarize_solutions(message)
    #     print("📦 解法統整完成，前 200 字：", reply[:200])
    #     save_query_context(chat_id, message, query_type, result_summary=reply[:200])
    #     return reply

    # 預設為 Semantic Query
    print("🔄 類型為語意查詢，開始檢索知識庫...")
    
    # ✅ 動態決定 top_k 筆數（預設 fallback=3）
    top_k = determine_top_k_with_llm(message, fallback=3)
    retrieved = search_knowledge_base(message, top_k=top_k)
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
