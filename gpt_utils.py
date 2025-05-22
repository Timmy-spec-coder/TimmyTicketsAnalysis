from SmartScoring import is_actionable_resolution
import aiohttp
import asyncio
import hashlib
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np

MAX_CONCURRENCY = 10
DEFAULT_MODEL_SOLUTION = "mistral"
DEFAULT_MODEL_SUMMARY = "phi3:mini"
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# ✅ 快取儲存位置
CACHE_DIR = "cache"
CACHE_FILE = os.path.join(CACHE_DIR, "semantic_cache.json")
MAX_CACHE_SIZE = 3000

# 統計用變數
cache_hit_count = 0
cache_total_queries = 0

# ✅ 確保資料夾存在
os.makedirs(CACHE_DIR, exist_ok=True)

# ✅ 載入語意模型
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ 載入快取資料
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        semantic_cache = json.load(f)
else:
    semantic_cache = []

# ✅ 產生 hash key 用於完全比對
def make_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# ✅ 語意快取查詢（含文字 hash + cosine）
# ✅ 語意快取查詢（含文字 hash + cosine）
def find_semantic_cache(text, threshold=0.9, source_id=""):
    global cache_hit_count, cache_total_queries
    cache_total_queries += 1
    key = make_hash(text)

    print(f"🔍 [Cache] 查找快取中... {source_id} hash={key[:8]} text='{text[:30]}'")

    for item in semantic_cache:
        if item.get("hash") == key:
            if item["response"] == "（AI 擷取失敗）":
                print(f"🚫 [Cache] 命中但為失敗快取（{source_id}），需送 GPT 再分析")
                return None
            cache_hit_count += 1
            print(f"🎯 [Cache] 完整命中！（{source_id}）")
            return item["response"]

    if len(semantic_cache) == 0:
        print(f"📭 [Cache] 無任何快取可比對（cache 空）（{source_id}）")
        return None

    try:
        query_vec = embedding_model.encode(text).astype(np.float32)
        all_vecs = np.array([item['embedding'] for item in semantic_cache], dtype=np.float32)
        sims = util.cos_sim(query_vec, all_vecs).flatten()
    except Exception as e:
        print(f"❌ [Cache] 語意比對時發生錯誤（{source_id}）：{e}")
        return None

    best_idx = int(np.argmax(sims))
    if sims[best_idx] > threshold:
        response = semantic_cache[best_idx]["response"]
        if response == "（AI 擷取失敗）":
            print(f"🚫 [Cache] 語意相似命中但為失敗快取（{sims[best_idx]:.3f}）（{source_id}）")
            return None
        cache_hit_count += 1
        print(f"🎯 [Cache] 語意相似命中！相似度={sims[best_idx]:.3f}（{source_id}）")
        return response

    print(f"❌ [Cache] 無命中，將送 GPT 擷取新資料（{source_id}）")
    return None




# ✅ 儲存新的快取紀錄
def add_to_semantic_cache(text, response):
    key = make_hash(text)
    emb = embedding_model.encode(text).tolist()
    semantic_cache.append({
        "hash": key,
        "input": text,
        "embedding": emb,
        "response": response,
        "createdAt": datetime.now().isoformat()
    })
    if len(semantic_cache) > MAX_CACHE_SIZE:
        semantic_cache.pop(0)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(semantic_cache, f, ensure_ascii=False, indent=2)
    print(f"💾 [Cache] 已儲存快取：hash={key[:8]} text='{text[:30]}'")

# 🧠 主功能：從段落中抽出解決建議句（含空值與快取）

def get_gpt_prompt_and_model(task="solution"):
    MAP_PATH = os.path.join("gpt_data", "gpt_prompt_map.json")
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get(task):
            prompt = config[task].get("prompt") or ""
            model = config[task].get("model") or ""
            # 檢查是否 prompt/model 缺失
            if not prompt or not model:
                print(f"⚠️ [PromptMap] {task} 的 prompt 或 model 欄位為空，已啟用預設值！")
            return prompt, model
        else:
            print(f"⚠️ [PromptMap] {task} 設定不存在，已啟用預設值！")
    except Exception as e:
        print(f"❌ [PromptMap] 讀取 {task} 設定失敗：{e}")
    # 回傳預設值
    if task == "solution":
        print("⚠️ [PromptMap] 使用 solution 預設 prompt/model")
        return "請從以下段落提取一個具體的行動建議", DEFAULT_MODEL_SOLUTION
    elif task == "ai_summary":
        print("⚠️ [PromptMap] 使用 ai_summary 預設 prompt/model")
        return "請用一句話描述事件是什麼", DEFAULT_MODEL_SUMMARY
    else:
        print(f"⚠️ [PromptMap] 未知用途 {task}，回傳空值")
        return "", ""



async def extract_resolution_suggestion(text, model= DEFAULT_MODEL_SOLUTION, source_id=""):
    if not isinstance(text, str) or not text.strip():
        return "（無原始描述）"

    ALWAYS_ANALYZE = True
    if not ALWAYS_ANALYZE and not is_actionable_resolution(text):
        print(f"⏭️ 無語意相近解法語氣，略過分析（{source_id}）：", text[:100])
        return "（未偵測到具體解法語氣，略過分析）"

    # 讀取目前的 prompt 與 model 設定
    custom_prompt, custom_model = get_gpt_prompt_and_model("solution")
    model = model or custom_model


    lines = text.strip().splitlines()
    text_trimmed = "\n".join(lines[:3])
    print(f"🔍 [GPT] 準備擷取解決建議：{text_trimmed[:30]}...（{source_id}）")

    cached = find_semantic_cache(text_trimmed, source_id=source_id)
    if cached:
        print(f"🎯 快取命中：略過 GPT 分析（{source_id}）")
        return cached

    prompt = f"{custom_prompt}\n---\n{text_trimmed}"
    try:
        result = await call_ollama_model_async(prompt, model)
        print(f"✅ GPT 第一次呼叫成功（{source_id}）")
        add_to_semantic_cache(text_trimmed, result)
        return result
    except Exception as e1:
        print(f"⚠️ GPT 第一次呼叫失敗（{source_id}）：{e1}")
        print(f"🔁 正在重新嘗試第 2 次 GPT 呼叫...（{source_id}）")

        await asyncio.sleep(2)
        try:
            result = await call_ollama_model_async(prompt, model)
            print(f"✅ GPT 第二次呼叫成功（{source_id}）")
            add_to_semantic_cache(text_trimmed, result)
            return result
        except Exception as e2:
            print(f"⛔ GPT 第二次呼叫也失敗（{source_id}）：{e2}")
            return "（AI 擷取失敗）"


async def extract_problem_with_custom_prompt(text, model=None, source_id=""):
    if not isinstance(text, str) or not text.strip():
        return "（無原始描述）"

    # 讀取目前的 prompt 與 model 設定
    custom_prompt, custom_model = get_gpt_prompt_and_model("ai_summary")
    model = model or custom_model

    lines = text.strip().splitlines()
    text_trimmed = "\n".join(lines[:3])
    print(f"🔍 [GPT] 準備擷取問題摘要：{text_trimmed[:30]}...（{source_id}）")

    cached = find_semantic_cache(text_trimmed, source_id=source_id)
    if cached:
        print(f"🎯 快取命中：略過 GPT 分析（{source_id}）")
        return cached

    prompt = f"{custom_prompt}\n---\n{text_trimmed}"

    try:
        result = await call_ollama_model_async(prompt, model)
        print(f"✅ GPT 第一次呼叫成功（{source_id}）")
        add_to_semantic_cache(text_trimmed, result)
        return result
    except Exception as e1:
        print(f"⚠️ GPT 第一次呼叫失敗（{source_id}）：{e1}")
        print(f"🔁 正在重新嘗試第 2 次 GPT 呼叫...（{source_id}）")
        await asyncio.sleep(2)
        try:
            result = await call_ollama_model_async(prompt, model)
            print(f"✅ GPT 第二次呼叫成功（{source_id}）")
            add_to_semantic_cache(text_trimmed, result)
            return result
        except Exception as e2:
            print(f"⛔ GPT 第二次呼叫也失敗（{source_id}）：{e2}")
            return "（AI 擷取失敗）"
# 🧠 主功能：擷取問題摘要（同樣支援 source_id）

# 📊 快取命中率報告
def print_cache_report():
    if cache_total_queries == 0:
        print("📊 本次未執行任何語意快取查詢。")
        return
    ratio = cache_hit_count / cache_total_queries * 100
    print(f"📊 快取命中 {cache_hit_count} / {cache_total_queries} 筆，命中率 {ratio:.1f}%")

# 🔧 非同步呼叫本地 Ollama API
async def call_ollama_model_async(prompt, model="phi3:mini", timeout=120):
    async with semaphore:
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 50,
                "temperature": 0.5
            }
        }

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                return result.get("response", "").strip()
