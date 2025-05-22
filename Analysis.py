# 匯入 Flask 框架及相關模組
from flask import Flask, request, jsonify, render_template, session, send_file
from gpt_utils import extract_resolution_suggestion
from gpt_utils import extract_problem_with_custom_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from collections import Counter
import umap
import hdbscan
# 匯入數學運算模組
import math
import json
# 匯入 pandas 用於處理 Excel 資料
import pandas as pd
# 匯入 os 模組處理檔案與路徑
import os
import shutil
# 匯入正則表達式模組
import re
import glob
# 匯入 webbrowser 用於開啟網頁
import webbrowser
import socket
# 匯入 traceback 用於錯誤追蹤
import traceback
# 匯入 Werkzeug 的工具函數確保檔案名稱安全
from werkzeug.utils import secure_filename
# ✅ 匯入語意分析模組
from SmartScoring import is_high_risk, is_escalated, is_multi_user, extract_keywords, recommend_solution, is_actionable_resolution, load_embeddings, load_examples_from_json
# ✅ 預先 encode 一筆資料以加速首次請求
from SmartScoring import bert_model  # 確保你有從 SmartScoring 載入模型
from SmartScoring import extract_cluster_name  # 匯入自定的 cluster 命名函式
from tqdm import tqdm
from sentence_transformers import util
# ✅ 匯入關鍵字抽取模組
from datetime import datetime
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime
import time
# --- 分群啟用條件（可依資料調整）---
import asyncio
import math

KMEANS_MIN_COUNT = 4         # 最少資料筆數
KMEANS_MIN_RANGE = 5.0       # 分數最大最小值差
KMEANS_MIN_STDDEV = 3.0      # 標準差下限


start = time.time()
print("🔥 預熱語意模型中...")
bert_model.encode("warmup")  # 預熱一次，避免第一次使用太慢
print(f"✅ 模型預熱完成，用時：{time.time() - start:.2f} 秒")

# 建立 Flask 應用
app = Flask(__name__)
# 設定應用的密鑰，用於 session 加密
app.secret_key = 'gwegweqgt22e'
# 設定 session 儲存方式為檔案系統
app.config['SESSION_TYPE'] = 'filesystem'

# ------------------------------------------------------------------------------
# 設定上傳資料夾與大小限制（10MB）
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 限制檔案大小為 10MB
ALLOWED_EXTENSIONS = {'xlsx'}  # 僅允許上傳 xlsx 檔案


basedir = os.path.abspath(os.path.dirname(__file__))  # 取得當前 app.py 的絕對目錄
# 確保上傳資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(basedir, 'json_data'), exist_ok=True)
os.makedirs(os.path.join(basedir, 'excel_result_Unclustered'), exist_ok=True)  # 新增未分群資料夾
os.makedirs(os.path.join(basedir, 'excel_result_Clustered'), exist_ok=True) # 新增分群資料夾



# 判斷是否允許的檔案格式
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------------------------------------------------------------------

# 定義函數：處理特殊值（如 NaN、None 等）
def safe_value(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return 0
    elif val is None:
        return None
    elif isinstance(val, str):
        return val
    else:
        return val

# ------------------------------------------------------------------------------




@app.route('/check-unclustered', methods=['GET'])
def check_unclustered_files():
    folder = 'excel_result_Unclustered'
    if not os.path.exists(folder):
        return jsonify({'exists': False}), 200
    files = [f for f in os.listdir(folder) if f.endswith('.xlsx')]
    return jsonify({'exists': len(files) > 0}), 200


@app.route('/clustered-files', methods=['GET'])
def list_clustered_files():
    clustered_folder = 'excel_result_Clustered'
    if not os.path.exists(clustered_folder):
        return jsonify({'files': []})

    pattern = re.compile(r"^Cluster-\[CI\].+_\[RC\].+_\[SC\].+\.xlsx$")
    files_info = []

    for f in os.listdir(clustered_folder):
        if not (f.endswith('.xlsx') and pattern.match(f)):
            continue

        filepath = os.path.join(clustered_folder, f)
        try:
            df = pd.read_excel(filepath)
            row_count = len(df)
        except Exception as e:
            print(f"❌ 無法讀取 {f}：{e}")
            row_count = 0

        files_info.append({
            'name': f,
            'rows': row_count
        })

    # ✅ 可選：依照 row 數降冪排序（最多的排前面）
    files_info.sort(key=lambda x: x['rows'], reverse=True)

    return jsonify({'files': files_info})


@app.route('/download-clustered', methods=['GET'])
def download_clustered_file():
    filename = request.args.get('file')
    path = os.path.join('excel_result_Clustered', filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': '找不到檔案'}), 404




# 根據分數判斷風險等級（支援 KMeans 分群）
kmeans_thresholds = None  # 全域變數，存儲 KMeans 分群門檻


def get_risk_level(score):
    global kmeans_thresholds
    level = ''

    if kmeans_thresholds and len(kmeans_thresholds) == 4:
        thresholds = sorted(kmeans_thresholds)
        if score >= thresholds[3]:
            level = '高風險'
        elif score >= thresholds[2]:
            level = '中風險'
        elif score >= thresholds[1]:
            level = '低風險'
        else:
            level = '忽略'
        print(f"📊 KMeans：impactScore: {score} → 分級：{level}（使用動態門檻）")
    else:
        if score >= 18:
            level = '高風險'
        elif score >= 12:
            level = '中風險'
        elif score >= 6:
            level = '低風險'
        else:
            level = '忽略'
        print(f"📊 固定門檻：impactScore: {score} → 分級：{level}")

    return level

# 在分析完成後自動設定 kmeans_thresholds
# （請放在 KMeans 分群完成後）
def set_kmeans_thresholds_from_centroids(centroids):
    global kmeans_thresholds
    kmeans_thresholds = sorted(centroids)
    print(f"✅ 已設定 KMeans 分群門檻（sorted）：{kmeans_thresholds}")

# ------------------------------------------------------------------------------


# ✅ 新增路由：處理所有 Unclustered Excel 檔案的分群與搬移
@app.route('/cluster-excel', methods=['POST'])
def cluster_excel():
    unclustered_dir = 'excel_result_Unclustered'
    clustered_dir = 'excel_result_Clustered'
    os.makedirs(clustered_dir, exist_ok=True)  # ✅ 確保 Clustered 資料夾存在

    files = [f for f in os.listdir(unclustered_dir) if f.endswith('_Unclustered.xlsx')]
    print(f"🔍 偵測到 {len(files)} 筆待分群檔案")

    for filename in files:
        uid = filename.replace('_Unclustered.xlsx', '')
        excel_path = os.path.join(unclustered_dir, filename)
        print(f"📂 處理檔案：{excel_path}")

        # 載入 Excel 並進行分群匯出
        df = pd.read_excel(excel_path)
        results = df.to_dict(orient='records')
        cluster_excel_export(results)  # ✅ 呼叫已定義的函式進行分群匯出

        # 搬移檔案到 Clustered 並改名
        clustered_path = os.path.join(clustered_dir, uid + '_Clustered.xlsx')
        shutil.move(excel_path, clustered_path)
        print(f"📁 已移動並改名：{clustered_path}")

    return jsonify({'message': f'已成功處理 {len(files)} 筆 Excel 檔案並完成分群'}), 200

# ------------------------------------------------------------------------------

def cluster_excel_export(results, export_dir="excel_result_Clustered"):
    def clean(text):
        return re.sub(r'[^\w\-_.]', '_', str(text).strip())[:30] or "Unknown"

    cluster_data = defaultdict(list)
    for r in results:
        config_item = r.get("configurationItem", "Unknown")
        role_component = r.get("roleComponent", "Unknown")
        subcategory = r.get("subcategory", "Unknown")
        cluster_key = f"{config_item}_{role_component}_{subcategory}"
        r['cluster'] = cluster_key
        cluster_data[cluster_key].append(r)

    os.makedirs(export_dir, exist_ok=True)

    for key, group in cluster_data.items():
        cluster_df = pd.DataFrame(group)

        try:
            config_item, role_component, subcategory = key.split('_', 2)
        except ValueError:
            config_item, role_component, subcategory = key, "Unknown", "Unknown"

        filename = f"{export_dir}/Cluster-[CI]{clean(config_item)}_[RC]{clean(role_component)}_[SC]{clean(subcategory)}.xlsx"

        if os.path.exists(filename):
            old_df = pd.read_excel(filename)
            cluster_df = pd.concat([old_df, cluster_df], ignore_index=True)

        cluster_df = cluster_df.sort_values(by="analysisTime", ascending=False)
        cluster_df.to_excel(filename, index=False)
        print(f"📁 已輸出：{filename}（共 {len(cluster_df)} 筆）")

        high_count = sum(1 for e in group if e.get('riskLevel') == '高風險')
        total = len(group)
        if total > 0 and (high_count / total) >= 0.5:
            print(f"🚨 預警：Cluster {key} 有 {high_count}/{total} 筆高風險事件")
    print("✅ 分群 Excel 檔案已儲存！")






# 用於同步 Flask 路由呼叫 async 分析邏輯
def analyze_excel(filepath, weights=None):
    return asyncio.run(analyze_excel_async(filepath, weights))


# 用於同步 Flask 路由呼叫 async 分析邏輯
async def analyze_excel_async(filepath, weights=None):
    start_time = time.time()
    default_weights = {
        'keyword': 5.0,
        'multi_user': 3.0,
        'escalation': 2.0,
        'config_item': 5.0,
        'role_component': 3.0,
        'time_cluster': 2.0
    }
    weights = {**default_weights, **(weights or {})}
    print(f"🟩 本次分析開始，將即時讀取三類語句 json 檔案...")
    # ⭐ 讀取語句和 embedding
    high_risk_examples, high_risk_embeddings = load_embeddings("high_risk")
    escalation_examples, escalation_embeddings = load_embeddings("escalate")
    multi_user_examples, multi_user_embeddings = load_embeddings("multi_user")

    df = pd.read_excel(filepath)
    component_counts = df['Role/Component'].value_counts()
    configuration_item_counts = df['Configuration item'].value_counts()
    configuration_item_max = configuration_item_counts.max()
    df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')
    analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 非同步處理
    tasks = [
        analyze_row_async(
            row, idx, df, weights, component_counts, configuration_item_counts, configuration_item_max, analysis_time,
            high_risk_examples, high_risk_embeddings,
            escalation_examples, escalation_embeddings,
            multi_user_examples, multi_user_embeddings)
        for idx, row in df.iterrows()
    ]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in results_raw if r and not isinstance(r, Exception)]

    # ✅ 防呆：沒有任何成功的結果就直接回傳避免崩潰
    if not results:
        print("⚠️ 所有資料都無法分析，請檢查欄位是否缺失")
        return {
            'data': [],
            'analysisTime': analysis_time
        }

    all_scores = [r['impactScore'] for r in results]
    score_range = max(all_scores) - min(all_scores)
    score_std = np.std(all_scores)

    print(f"📈 分群判斷指標：count={len(all_scores)}, range={score_range:.2f}, stddev={score_std:.2f}")

    if (
        len(all_scores) >= KMEANS_MIN_COUNT and
        score_range >= KMEANS_MIN_RANGE and
        score_std >= KMEANS_MIN_STDDEV
    ):
        kmeans = KMeans(n_clusters=4, random_state=42)
        labels = kmeans.fit_predict(np.array(all_scores).reshape(-1, 1))
        centroids = kmeans.cluster_centers_.flatten()
        set_kmeans_thresholds_from_centroids(centroids)
        print(f"📊 KMeans 分群標籤：{labels}")
        label_map = {}
        for i, idx in enumerate(np.argsort(centroids)[::-1]):
            label_map[idx] = ['高風險', '中風險', '低風險', '忽略'][i]
        for i, r in enumerate(results):
            r['riskLevel'] = label_map[labels[i]]
        print(f"📌 KMeans 分群中心：{sorted(centroids, reverse=True)}")
    else:
        print("⚠️ 不啟用 KMeans，改用固定門檻分級")
        for r in results:
            r['riskLevel'] = get_risk_level(r['impactScore'])

    total_time = time.time() - start_time
    avg_time = total_time / len(results)

    print(f"\n🎯 所有分析總耗時：{total_time:.2f} 秒")
    print(f"📊 單筆平均耗時：{avg_time:.2f} 秒")
    print("\n✅ 所有資料分析完成！")
    return {
        'data': results,
        'analysisTime': analysis_time
    }






async def analyze_row_async(row, idx, df, weights, component_counts, configuration_item_counts, configuration_item_max, analysis_time,     
    high_risk_examples, high_risk_embeddings,
    escalation_examples, escalation_embeddings,
    multi_user_examples, multi_user_embeddings):
    print(f"[分析 Row#{idx+1}] 本次用的高風險語句數：{len(high_risk_examples)}，倒數兩句：{high_risk_examples[-2:] if high_risk_examples else '空'}")
    print(f"[分析 Row#{idx+1}] 本次用的升級語句數：{len(escalation_examples)}，倒數兩句：{escalation_examples[-2:] if escalation_examples else '空'}")
    print(f"[分析 Row#{idx+1}] 本次用的影響多使用者語句數：{len(multi_user_examples)}，倒數兩句：{multi_user_examples[-2:] if multi_user_examples else '空'}")
    try:
        # 原始欄位保留
        description_text = row.get('Description', 'not filled')
        short_description_text = row.get('Short description', 'not filled')
        close_note_text = row.get('Close notes', 'not filled')

        # 字串清理（保留變數命名）
        desc = str(description_text).strip()
        short_desc = str(short_description_text).strip()
        close_notes = str(close_note_text).strip()

        # 若全部內容皆為空，直接跳過此筆
        if not (desc or short_desc or close_notes):
            print(f"⚠️ 第 {idx+1} 筆內容全為空白，略過分析")
            return None

        resolution_text = f"{desc}\n{short_desc}\n{close_notes}".strip()
        if len(resolution_text) > 10000:
            print(f"🟡 [Row#{idx+1}] resolution_text > 3000，使用 short_desc + close_notes")
            resolution_text = f"{short_desc}\n{close_notes}".strip()
            if len(resolution_text) > 10000:
                print(f"🔴 [Row#{idx+1}] short_desc + close_notes > 3000，只用 close_notes")
                resolution_text = close_notes.strip()
        else:
            print(f"🟢 [Row#{idx+1}] resolution_text 使用 desc + short_desc + close_notes")


        keyword_score = is_high_risk(short_desc, high_risk_examples, high_risk_embeddings)
        user_impact_score = is_multi_user(desc, multi_user_examples, multi_user_embeddings)
        escalation_score = is_escalated(close_notes, escalation_examples, escalation_embeddings)

        config_raw = configuration_item_counts.get(row.get('Configuration item'), 0)
        configuration_item_freq = config_raw / configuration_item_max if configuration_item_max > 0 else 0

        role_comp = row.get('Role/Component', 'not filled')
        count = component_counts.get(role_comp, 0)
        role_component_freq = 3 if count >= 5 else 2 if count >= 3 else 1 if count == 2 else 0

        this_time = row.get('Opened', 'not filled')
        if pd.isnull(this_time):
            time_cluster_score = 1
        else:
            others = df[df['Role/Component'] == role_comp]
            close_events = others[(others['Opened'] >= this_time - pd.Timedelta(hours=24)) &
                                  (others['Opened'] <= this_time + pd.Timedelta(hours=24))]
            count_cluster = len(close_events)
            time_cluster_score = 3 if count_cluster >= 3 else 2 if count_cluster == 2 else 1

        severity_score = round(
            keyword_score * weights['keyword'] +
            user_impact_score * weights['multi_user'] +
            escalation_score * weights['escalation'], 2
        )
        frequency_score = round(
            configuration_item_freq * weights['config_item'] +
            role_component_freq * weights['role_component'] +
            time_cluster_score * weights['time_cluster'], 2
        )
        impact_score = round(math.sqrt(severity_score**2 + frequency_score**2), 2)
        risk_level = get_risk_level(impact_score)

        # ==== 判斷 summary 輸入長度 ====
        summary_input = f"{short_desc}\n{desc}".strip()
        if len(summary_input) > 8000:
            print(f"🟡 [Row#{idx+1}] summary_input > 2000，使用 short_desc + close_notes")
            summary_input = f"{short_desc}\n{close_notes}".strip()
            if len(summary_input) > 8000:
                print(f"🔴 [Row#{idx+1}] short_desc + close_notes > 2000，只用 short_desc")
                summary_input = short_desc.strip()
        else:
            print(f"🟢 [Row#{idx+1}] summary_input 使用 short_desc + desc")



        # GPT 處理允許失敗
        try:
            ai_suggestion, ai_summary = await asyncio.gather(
                extract_resolution_suggestion(resolution_text, source_id=f"Row#{idx+1}"),
                extract_problem_with_custom_prompt(summary_input, source_id=f"Row#{idx+1}")
            )

        except Exception as e:
            print(f"⚠️ GPT 擷取失敗：{e}")
            ai_suggestion = "（AI 擷取失敗）"
            ai_summary = "（AI 擷取失敗）"

        recommended = recommend_solution(short_desc)
        keywords = extract_keywords(short_desc)

        return {
            'id': safe_value(row.get('Incident') or row.get('Number')),
            'configurationItem': safe_value(row.get('Configuration item')),
            'roleComponent': safe_value(row.get('Role/Component')),
            'subcategory': safe_value(row.get('Subcategory')),
            'aiSummary': safe_value(ai_summary),
            'originalShortDescription': safe_value(short_desc),
            'originalDescription': safe_value(desc),
            'severityScore': safe_value(severity_score),
            'frequencyScore': safe_value(frequency_score),
            'impactScore': safe_value(impact_score),
            'severityScoreNorm': round(severity_score / 10, 2),
            'frequencyScoreNorm': round(frequency_score / 20, 2),
            'impactScoreNorm': round(impact_score / 30, 2),
            'riskLevel': risk_level,
            'solution': safe_value(ai_suggestion or '無提供解法'),
            'location': safe_value(row.get('Location')),
            'analysisTime': analysis_time,
            'weights': {k: round(v / 10, 2) for k, v in weights.items()},
        }

    except Exception as e:
        print(f"❌ 分析第 {idx + 1} 筆失敗：", e)
        return None










SENTENCE_DIR = os.path.join("data", "sentences")
os.makedirs(SENTENCE_DIR, exist_ok=True)

def get_file_path(tag):
    return os.path.join(SENTENCE_DIR, f"{tag}.json")

@app.route("/get-sentence-db")
def get_sentence_db():
    result = []
    for tag in ["high_risk", "escalate", "multi_user"]:
        path = get_file_path(tag)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                sentences = json.load(f)
                for entry in sentences:
                    result.append({"text": entry["text"], "tag": tag})
    return jsonify(result)

@app.route("/save-sentence-db", methods=["POST"])
def save_sentence():
    new_entry = request.get_json()
    tag = new_entry.get("tag")
    if tag not in ["high_risk", "escalate", "multi_user"]:
        return jsonify({"message": "invalid tag"}), 400

    path = get_file_path(tag)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    if any(d['text'] == new_entry['text'] for d in data):
        return jsonify({"message": "duplicate"}), 409

    data.append({"text": new_entry['text']})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return get_sentence_db()

@app.route("/delete-sentence", methods=["POST"])
def delete_sentence():
    req = request.get_json()
    tag = req.get("tag")
    text = req.get("text")
    if tag not in ["high_risk", "escalate", "multi_user"] or not text:
        return jsonify({"message": "invalid input"}), 400

    path = get_file_path(tag)
    if not os.path.exists(path):
        return jsonify({"message": "not found"}), 404

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    new_data = [d for d in data if d['text'] != text]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    return get_sentence_db()

@app.route("/edit-sentence", methods=["POST"])
def edit_sentence():
    req = request.get_json()
    tag = req.get("tag")
    old_text = req.get("oldText")
    new_text = req.get("newText")

    if tag not in ["high_risk", "escalate", "multi_user"] or not old_text or not new_text:
        return jsonify({"message": "invalid input"}), 400

    path = get_file_path(tag)
    if not os.path.exists(path):
        return jsonify({"message": "not found"}), 404

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    updated = False
    for d in data:
        if d['text'] == old_text:
            d['text'] = new_text
            updated = True
            break

    if not updated:
        return jsonify({"message": "not found"}), 404

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return get_sentence_db()








GPT_DATA_DIR = "gpt_data"
PROMPT_FILE = os.path.join(GPT_DATA_DIR, "gpt_prompts.json")
MAP_FILE = os.path.join(GPT_DATA_DIR, "gpt_prompt_map.json")

os.makedirs(GPT_DATA_DIR, exist_ok=True)

def read_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return default if default is not None else {}

def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

@app.route('/get-gpt-prompts')
def get_gpt_prompts():
    all_data = read_json(PROMPT_FILE, {})
    # 包成 {用途: {"prompts": [...]} }
    wrapped = {k: {"prompts": v if isinstance(v, list) else [v]} for k, v in all_data.items()}
    return jsonify(wrapped)

@app.route('/save-gpt-prompt', methods=['POST'])
def save_gpt_prompt():
    """
    新增一筆 prompt 到某個分類。body: { "task": "solution", "prompt": "xxx" }
    """
    data = request.get_json()
    task = data.get('task')
    prompt = data.get('prompt', '').strip()

    if not task or not prompt:
        return jsonify(success=False, message='❌ 缺少 task 或 prompt'), 400

    all_prompts = read_json(PROMPT_FILE, {})
    prompt_list = all_prompts.get(task, [])
    if not isinstance(prompt_list, list):
        prompt_list = [prompt_list]

    if prompt not in prompt_list:
        prompt_list.append(prompt)
    else:
        return jsonify(success=False, message='⚠️ 該 prompt 已存在'), 409

    all_prompts[task] = prompt_list
    write_json(PROMPT_FILE, all_prompts)

    return jsonify(success=True, allPrompts=all_prompts)

@app.route('/delete-gpt-prompt', methods=['POST'])
def delete_gpt_prompt():
    """
    支援刪除分類或分類下的單一句子。
    body: { "task": "solution", "prompt": "xxx" } 或只給 task 代表整類刪除。
    """
    data = request.get_json()
    task = data.get('task')
    prompt = data.get('prompt', '').strip()

    all_prompts = read_json(PROMPT_FILE, {})

    if task not in all_prompts:
        return jsonify(success=False, message=f'找不到用途 {task}'), 404

    if prompt:
        prompt_list = all_prompts[task]
        if prompt in prompt_list:
            prompt_list.remove(prompt)
            if prompt_list:
                all_prompts[task] = prompt_list
            else:
                del all_prompts[task]
        else:
            return jsonify(success=False, message='找不到該 prompt'), 404
    else:
        del all_prompts[task]  # 刪整類

    write_json(PROMPT_FILE, all_prompts)

    # 刪掉 mapping 中的對應
    mapping = read_json(MAP_FILE, {})
    if task in mapping:
        del mapping[task]
        write_json(MAP_FILE, mapping)

    return jsonify(success=True, allPrompts=all_prompts)

@app.route('/get-gpt-prompt-map')
def get_gpt_prompt_map():
    return jsonify(read_json(MAP_FILE, {}))

@app.route('/save-gpt-prompt-map', methods=['POST'])
def save_gpt_prompt_map():
    data = request.get_json()

    solution_prompt = data.get("solution")
    summary_prompt = data.get("ai_summary")
    models = data.get("models", {})

    new_mapping = {
        "solution": {
            "prompt": solution_prompt,
            "model": models.get("solution", "")
        },
        "ai_summary": {
            "prompt": summary_prompt,
            "model": models.get("ai_summary", "")
        }
    }

    write_json(MAP_FILE, new_mapping)
    return jsonify(success=True, mapping=new_mapping)

def get_prompt_for_use(use_type):
    mapping = read_json(MAP_FILE, {})
    all_prompts = read_json(PROMPT_FILE, {})
    mapped_key = mapping.get(use_type, use_type)
    prompt_list = all_prompts.get(mapped_key, [])
    if isinstance(prompt_list, list):
        return {'prompt': prompt_list[0] if prompt_list else '', 'model': mapping.get(mapped_key, {}).get("model", "")}
    return {'prompt': prompt_list, 'model': mapping.get(mapped_key, {}).get("model", "")}

# ========== 用法範例 ==========
# prompt_info = get_prompt_for_use("solution")
# print(prompt_info.get("model"), prompt_info.get("prompt"))












# ------------------------------------------------------------------------------


# # 分析 Excel 資料的主邏輯 
# def analyze_excel(filepath, weights=None):

#     start_time = time.time()
#     default_weights = {
#         'keyword': 5.0,
#         'multi_user': 3.0,
#         'escalation': 2.0,
#         'config_item': 5.0,
#         'role_component': 3.0,
#         'time_cluster': 2.0
#     }
#     weights = {**default_weights, **(weights or {})}
#     print("🎛️ 使用中的權重設定：", weights)

#     df = pd.read_excel(filepath)
#     print(f"📊 共讀取 {len(df)} 筆資料\n")

#     component_counts = df['Role/Component'].value_counts()
#     df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')
#     configuration_item_counts = df['Configuration item'].value_counts()
#     configuration_item_max = configuration_item_counts.max()
#     analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     def analyze_row(row, idx):
#         try:
#             description_text = row.get('Description', 'not filled')
#             short_description_text = row.get('Short description', 'not filled')
#             close_note_text = row.get('Close notes', 'not filled')

#             keyword_score = is_high_risk(short_description_text)
#             user_impact_score = is_multi_user(description_text)
#             escalation_score = is_escalated(close_note_text)

#             config_raw = configuration_item_counts.get(row.get('Configuration item'), 0)
#             configuration_item_freq = config_raw / configuration_item_max if configuration_item_max > 0 else 0

#             role_comp = row.get('Role/Component', 'not filled')
#             count = component_counts.get(role_comp, 0)
#             role_component_freq = 3 if count >= 5 else 2 if count >= 3 else 1 if count == 2 else 0

#             this_time = row.get('Opened', 'not filled')
#             if pd.isnull(this_time):
#                 time_cluster_score = 1
#             else:
#                 others = df[df['Role/Component'] == role_comp]
#                 close_events = others[(others['Opened'] >= this_time - pd.Timedelta(hours=24)) &
#                                       (others['Opened'] <= this_time + pd.Timedelta(hours=24))]
#                 count_cluster = len(close_events)
#                 time_cluster_score = 3 if count_cluster >= 3 else 2 if count_cluster == 2 else 1

#             severity_score = round(
#                 keyword_score * weights['keyword'] +
#                 user_impact_score * weights['multi_user'] +
#                 escalation_score * weights['escalation'], 2
#             )

#             frequency_score = round(
#                 configuration_item_freq * weights['config_item'] +
#                 role_component_freq * weights['role_component'] +
#                 time_cluster_score * weights['time_cluster'], 2
#             )

#             impact_score = round(math.sqrt(severity_score**2 + frequency_score**2), 2)
#             risk_level = get_risk_level(impact_score)

#             desc = str(row.get('Description', "")).strip()
#             short_desc = str(row.get('Short description', "")).strip()
#             close_notes = str(row.get('Close notes', "")).strip()
#             resolution_text = f"{desc}\n{short_desc}\n{close_notes}".strip()
#             ai_suggestion = extract_resolution_suggestion(resolution_text)
#             ai_summary = extract_problem_with_custom_prompt(f"{short_desc}\n{desc}".strip())
#             recommended = recommend_solution(short_description_text)
#             keywords = extract_keywords(short_description_text)

#             return {
#                 'id': safe_value(row.get('Incident') or row.get('Number')),
#                 'configurationItem': safe_value(row.get('Configuration item')),
#                 'roleComponent': safe_value(row.get('Role/Component')),
#                 'subcategory': safe_value(row.get('Subcategory')),
#                 'aiSummary': safe_value(ai_summary),
#                 'originalShortDescription': safe_value(short_desc),
#                 'originalDescription': safe_value(desc),
#                 'severityScore': safe_value(severity_score),
#                 'frequencyScore': safe_value(frequency_score),
#                 'impactScore': safe_value(impact_score),
#                 'severityScoreNorm': round(severity_score / 10, 2),
#                 'frequencyScoreNorm': round(frequency_score / 20, 2),
#                 'impactScoreNorm': round(impact_score / 30, 2),
#                 'riskLevel': risk_level,
#                 'solution': safe_value(ai_suggestion or '無提供解法'),
#                 'location': safe_value(row.get('Location')),
#                 'analysisTime': analysis_time,
#                 'weights': {k: round(v / 10, 2) for k, v in weights.items()},
#             }

#         except Exception as e:
#             print(f"❌ 分析第 {idx+1} 筆失敗：", e)
#             return None

#     # ✅ 非同步處理所有 row
#     results = []
#     per_row_times = []
#     with ThreadPoolExecutor(max_workers=8) as executor:
#         futures = {}
#         for idx, row in df.iterrows():
#             futures[executor.submit(analyze_row, row, idx)] = idx

#         for future in tqdm(as_completed(futures), total=len(futures), desc="📊 非同步分析中"):
#             idx = futures[future]
#             t0 = time.time()
#             res = future.result()
#             t1 = time.time()
#             elapsed = t1 - t0
#             per_row_times.append(elapsed)

#             if res:
#                 results.append(res)
#             print(f"⏱️ 第 {idx + 1} 筆：{elapsed:.2f} 秒完成")


#     # ✅ KMeans 分群（略）
#     # 可依照你原本的邏輯套用 KMeans，如：
#     # ⬇⬇⬇ KMeans 分群邏輯（支援三條件） ⬇⬇⬇
#     all_scores = [r['impactScore'] for r in results]
#     score_range = max(all_scores) - min(all_scores)
#     score_std = np.std(all_scores)

#     print(f"📈 分群判斷指標：count={len(all_scores)}, range={score_range:.2f}, stddev={score_std:.2f}")

#     if (
#         len(all_scores) >= KMEANS_MIN_COUNT and
#         score_range >= KMEANS_MIN_RANGE and
#         score_std >= KMEANS_MIN_STDDEV
#     ):
#         kmeans = KMeans(n_clusters=4, random_state=42)
#         labels = kmeans.fit_predict(np.array(all_scores).reshape(-1, 1))
#         centroids = kmeans.cluster_centers_.flatten()
#         set_kmeans_thresholds_from_centroids(centroids)
#         print(f"📊 KMeans 分群標籤：{labels}")
#         label_map = {}
#         for i, idx in enumerate(np.argsort(centroids)[::-1]):
#             label_map[idx] = ['高風險', '中風險', '低風險', '忽略'][i]
#         for i, r in enumerate(results):
#             r['riskLevel'] = label_map[labels[i]]
#         print(f"📌 KMeans 分群中心：{sorted(centroids, reverse=True)}")
#     else:
#         print("⚠️ 不啟用 KMeans，改用固定門檻分級")
#         for r in results:
#             r['riskLevel'] = get_risk_level(r['impactScore'])
#     # ⬆⬆⬆ 分群邏輯結束 ⬆⬆⬆


#     total_time = time.time() - start_time
#     avg_time = sum(per_row_times) / len(per_row_times) if per_row_times else 0
#     print(f"\n🎯 所有分析總耗時：{total_time:.2f} 秒")
#     print(f"📊 單筆平均耗時：{avg_time:.2f} 秒")

#     print("\n✅ 所有資料分析完成！")
#     return {
#         'data': results,
#         'analysisTime': analysis_time
#     }


# ------------------------------------------------------------------------------









# 定義首頁路由
@app.route('/')
def index():
    return render_template('FrontEnd.html')  # 渲染首頁模板

# 定義結果頁面路由
@app.route('/result')
def result_page():
    data = session.get('analysis_data', [])  # 從 session 取得分析結果
    return render_template('result.html', data=data)  # 渲染結果頁面

# 定義歷史紀錄頁面路由
@app.route('/history')
def history_page():
    return render_template('history.html')  # 渲染歷史紀錄頁面


@app.route('/generate_cluster')
def generate_cluster_page():
    return render_template('generate_cluster.html')  # 渲染生成分群頁面

@app.route("/manual_input")
def manual_input_page():
    return render_template("manual_input.html")

@app.route("/gpt_prompt")
def gpt_prompt_page():
    return render_template("gpt_prompt.html")


# ------------------------------------------------------------------------------




@app.route('/ping')
def ping():
    return "pong", 200


# 定義檔案上傳路由
@app.route('/upload', methods=['POST'])
def upload_file():
    print("📥 收到上傳請求")  # 紀錄請求



    if 'file' not in request.files:  # 檢查是否有檔案欄位
        print("❌ 沒有 file 欄位")
        return jsonify({'error': '沒有找到檔案欄位'}), 400

    file = request.files['file']  # 取得檔案
    if file.filename == '':  # 檢查檔案名稱是否為空
        print("⚠️ 檔案名稱為空")
        return jsonify({'error': '未選擇檔案'}), 400

    if not allowed_file(file.filename):  # 檢查檔案格式是否允許
        print("⚠️ 檔案類型不符")
        return jsonify({'error': '請上傳 .xlsx 檔案'}), 400
        
    # 接收自訂權重
    weights_raw = request.form.get('weights')
    if not weights_raw:
        print("ℹ️ 未提供自訂權重，使用預設值分析")

    weights = None
    if weights_raw:
        try:
            weights = json.loads(weights_raw)
            print("📥 收到權重設定：", weights)
        except Exception as e:
            print(f"⚠️ 權重解析失敗：{e}")
            return jsonify({'error': '權重解析失敗'}), 400
        
    # 產生時間戳記與檔名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    uid = f"result_{timestamp}" # 例如 result_20250423_152301 分析結果檔名稱
    original_filename = f"original_{timestamp}.xlsx" # 例如 original_20250423_152301.xlsx 原始黨名稱
    original_path = os.path.join('uploads', original_filename)



    try:
        file.save(original_path)  # 儲存原始檔案
        print(f" 原始檔已儲存：{original_path}")
    except Exception as e:
        return jsonify({'error': f'儲存原始檔失敗：{str(e)}'}), 500

    try:
        analysis_result = analyze_excel(original_path, weights=weights)
        results = analysis_result['data']  # 取得分析結果


        save_analysis_files(analysis_result, uid)  # 儲存分析結果檔案

        print(f"✅ 分析完成，共 {len(results)} 筆")
        session['analysis_data'] = results  # 儲存分析結果到 session
        return jsonify({'data': results, 'uid': uid, 'weights': weights}), 200


    


    except Exception as e:
        print(f"❌ 分析時發生錯誤：{e}")
        traceback.print_exc()  # 印出完整錯誤堆疊
        return jsonify({'error': str(e)}), 500
    
def save_analysis_files(result, uid):
    os.makedirs('json_data', exist_ok=True)
    os.makedirs('excel_result_Unclustered', exist_ok=True)  # ✅ 使用新的資料夾

    # 儲存 JSON
    json_path = os.path.join(basedir, 'json_data', f"{uid}.json")
    print(f"📝 預計儲存 JSON：{json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("✅ JSON 檔案已寫入成功")

    # 儲存分析報表 Excel（只儲存 result['data']）
    df = pd.DataFrame(result['data'])
    # ✅ 儲存到 Unclustered 資料夾並加上 Unclustered 後綴
    excel_filename = f"{uid}_Unclustered.xlsx"
    excel_path = os.path.join(basedir, 'excel_result_Unclustered', excel_filename)    
    df.to_excel(excel_path, index=False)

    # 確認 JSON 檔案是否寫入成功
    if os.path.exists(json_path):
        print("✅ JSON 檔案已成功儲存")
    else:
        print("❌ JSON 檔案儲存失敗！")

    print(f"✅ 分析報表已儲存：{excel_path}")
    print("📁 JSON 絕對路徑：", os.path.abspath(json_path))
    print("📁 Excel 絕對路徑：", os.path.abspath(excel_path))
    timestamp = uid.replace("result_", "")
    original_excel_path = os.path.abspath(os.path.join(basedir, 'uploads', f"original_{timestamp}.xlsx"))

    if os.path.exists(original_excel_path):
        print("📁 原始檔絕對路徑：", original_excel_path)
    else:
        print("⚠️ 找不到原始 Excel 路徑！")


@app.route('/get-results')
def get_results():
    folder = 'json_data'
    results = []
    first_weights = {}

    if not os.path.exists(folder):
        return jsonify({'error': f'資料夾不存在：{folder}'}), 404

    # 🔄 讀取所有檔案，找出最新的那份分析檔
    sorted_files = sorted(
        [f for f in os.listdir(folder) if f.endswith('.json')],
        reverse=True  # 最後面最新
    )

    for filename in sorted_files:
        filepath = os.path.join(folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, dict) and 'data' in content:
                    results.extend(content['data'])
                    if not first_weights and 'weights' in content:
                        first_weights = content['weights']
                elif isinstance(content, list):
                    results.extend(content)
        except Exception as e:
            print(f"❌ 錯誤讀取 {filename}：{e}")

    return jsonify({
        'data': results,
        'weights': first_weights  # ✅ 確保傳出這個欄位
    })










# ✅ JSON 預覽路由：提供 `/get-json?file=xxxx.json`
@app.route('/get-json', methods=['GET'])
def get_json_file():
    filename = request.args.get('file')  # e.g., result_20250423_152301.json
    if not filename:
        return jsonify({'error': '缺少 file 參數'}), 400

    json_path = os.path.join('json_data', filename)
    if os.path.exists(json_path):
        return send_file(json_path, as_attachment=False)
    else:
        return jsonify({'error': '找不到對應的 JSON 檔案'}), 404


@app.route('/download-excel', methods=['GET'])
def download_excel_file():
    uid = request.args.get('uid')  # e.g., result_20250508_203611
    if not uid:
        return jsonify({'error': '缺少 uid 參數'}), 400

    # 先檢查 Clustered
    clustered_path = os.path.join('excel_result_Clustered', f"{uid}_Clustered.xlsx")
    if os.path.exists(clustered_path):
        return send_file(clustered_path, as_attachment=True)

    # 再檢查 Unclustered
    unclustered_path = os.path.join('excel_result_Unclustered', f"{uid}_Unclustered.xlsx")
    if os.path.exists(unclustered_path):
        return send_file(unclustered_path, as_attachment=True)

    return jsonify({'error': f'找不到 {uid} 對應的 Excel 檔案'}), 404


@app.route('/download-original', methods=['GET'])
def download_original_excel():
    uid = request.args.get('uid')  # uid = result_20250423_152301
    if not uid:
        return jsonify({'error': '缺少 uid 參數'}), 400

    # 取出對應的時間戳
    timestamp = uid.replace('result_', '')
    original_filename = f'original_{timestamp}.xlsx'
    original_path = os.path.join('uploads', original_filename)

    if os.path.exists(original_path):
        return send_file(original_path, as_attachment=True)
    else:
        return jsonify({'error': '找不到對應的原始檔案'}), 404


# ------------------------------------------------------------------------------

# 定義檔案列表路由
@app.route('/files', methods=['GET'])
def get_file_list():
    files = os.listdir(UPLOAD_FOLDER)  # 列出上傳資料夾中的檔案
    return jsonify({'files': files}), 200

# ------------------------------------------------------------------------------

# 定義執行動作的路由
@app.route('/perform-action', methods=['POST'])
def perform_action():
    data = request.json  # 取得 JSON 資料
    action = data.get('action')  # 取得動作名稱

    if action == 'start':  # 如果動作是 'start'
        result = "Server received 'start' action and performed the task."
        print(result)
        return jsonify({'status': 'success', 'message': result}), 200
    else:  # 如果是未知動作
        result = f"Server received unknown action: {action}"
        print(result)
        return jsonify({'status': 'error', 'message': 'Unknown action!'}), 400

# ------------------------------------------------------------------------------

def is_flask_running():
    """檢查 Flask (127.0.0.1:5000) 是否已啟動"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 5000))
        s.shutdown(socket.SHUT_RDWR)
        return True
    except:
        return False
    finally:
        s.close()

if __name__ == "__main__":
    # 判斷 Flask 是否已經有服務
    if not is_flask_running():
        print("🌐 開啟瀏覽器 http://127.0.0.1:5000")
        webbrowser.open("http://127.0.0.1:5000")
    else:
        print("⚠️ Flask 已在運作，不重複開啟瀏覽器")
    app.run(debug=True, use_reloader=True)











# async def analyze_excel_async(filepath, weights=None):
#     start_time = time.time()
#     default_weights = {
#         'keyword': 5.0,
#         'multi_user': 3.0,
#         'escalation': 2.0,
#         'config_item': 5.0,
#         'role_component': 3.0,
#         'time_cluster': 2.0
#     }
#     weights = {**default_weights, **(weights or {})}

#     df = pd.read_excel(filepath)
#     component_counts = df['Role/Component'].value_counts()
#     configuration_item_counts = df['Configuration item'].value_counts()
#     configuration_item_max = configuration_item_counts.max()
#     df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')
#     analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     # 非同步處理每筆資料
#     tasks = [
#         analyze_row_async(row, idx, df, weights, component_counts, configuration_item_counts, configuration_item_max, analysis_time)
#         for idx, row in df.iterrows()
#     ]
#     results_raw = await asyncio.gather(*tasks)
#     results = [r for r in results_raw if r]

#     # ✅ 分群邏輯（照原本邏輯即可）
#     all_scores = [r['impactScore'] for r in results]
#     score_range = max(all_scores) - min(all_scores)
#     score_std = np.std(all_scores)

#     print(f"📈 分群判斷指標：count={len(all_scores)}, range={score_range:.2f}, stddev={score_std:.2f}")

#     if (
#         len(all_scores) >= KMEANS_MIN_COUNT and
#         score_range >= KMEANS_MIN_RANGE and
#         score_std >= KMEANS_MIN_STDDEV
#     ):
#         kmeans = KMeans(n_clusters=4, random_state=42)
#         labels = kmeans.fit_predict(np.array(all_scores).reshape(-1, 1))
#         centroids = kmeans.cluster_centers_.flatten()
#         set_kmeans_thresholds_from_centroids(centroids)
#         print(f"📊 KMeans 分群標籤：{labels}")
#         label_map = {}
#         for i, idx in enumerate(np.argsort(centroids)[::-1]):
#             label_map[idx] = ['高風險', '中風險', '低風險', '忽略'][i]
#         for i, r in enumerate(results):
#             r['riskLevel'] = label_map[labels[i]]
#         print(f"📌 KMeans 分群中心：{sorted(centroids, reverse=True)}")
#     else:
#         print("⚠️ 不啟用 KMeans，改用固定門檻分級")
#         for r in results:
#             r['riskLevel'] = get_risk_level(r['impactScore'])

#     total_time = time.time() - start_time
#     avg_time = total_time / len(results) if results else 0

#     print(f"\n🎯 所有分析總耗時：{total_time:.2f} 秒")
#     print(f"📊 單筆平均耗時：{avg_time:.2f} 秒")

#     print("\n✅ 所有資料分析完成！")
#     return {
#         'data': results,
#         'analysisTime': analysis_time
#     }







# async def analyze_row_async(row, idx, df, weights, component_counts, configuration_item_counts, configuration_item_max, analysis_time):
#     try:
#         description_text = row.get('Description', 'not filled')
#         short_description_text = row.get('Short description', 'not filled')
#         close_note_text = row.get('Close notes', 'not filled')

#         keyword_score = is_high_risk(short_description_text)
#         user_impact_score = is_multi_user(description_text)
#         escalation_score = is_escalated(close_note_text)

#         config_raw = configuration_item_counts.get(row.get('Configuration item'), 0)
#         configuration_item_freq = config_raw / configuration_item_max if configuration_item_max > 0 else 0

#         role_comp = row.get('Role/Component', 'not filled')
#         count = component_counts.get(role_comp, 0)
#         role_component_freq = 3 if count >= 5 else 2 if count >= 3 else 1 if count == 2 else 0

#         this_time = row.get('Opened', 'not filled')
#         if pd.isnull(this_time):
#             time_cluster_score = 1
#         else:
#             others = df[df['Role/Component'] == role_comp]
#             close_events = others[(others['Opened'] >= this_time - pd.Timedelta(hours=24)) &
#                                   (others['Opened'] <= this_time + pd.Timedelta(hours=24))]
#             count_cluster = len(close_events)
#             time_cluster_score = 3 if count_cluster >= 3 else 2 if count_cluster == 2 else 1

#         severity_score = round(
#             keyword_score * weights['keyword'] +
#             user_impact_score * weights['multi_user'] +
#             escalation_score * weights['escalation'], 2
#         )
#         frequency_score = round(
#             configuration_item_freq * weights['config_item'] +
#             role_component_freq * weights['role_component'] +
#             time_cluster_score * weights['time_cluster'], 2
#         )
#         impact_score = round(math.sqrt(severity_score**2 + frequency_score**2), 2)
#         risk_level = get_risk_level(impact_score)

#         desc = str(description_text).strip()
#         short_desc = str(short_description_text).strip()
#         close_notes = str(close_note_text).strip()
#         resolution_text = f"{desc}\n{short_desc}\n{close_notes}".strip()

#         ai_suggestion, ai_summary = await asyncio.gather(
#             extract_resolution_suggestion(resolution_text),
#             extract_problem_with_custom_prompt(f"{short_desc}\n{desc}".strip())
#         )

#         recommended = recommend_solution(short_description_text)
#         keywords = extract_keywords(short_description_text)

#         return {
#             'id': safe_value(row.get('Incident') or row.get('Number')),
#             'configurationItem': safe_value(row.get('Configuration item')),
#             'roleComponent': safe_value(row.get('Role/Component')),
#             'subcategory': safe_value(row.get('Subcategory')),
#             'aiSummary': safe_value(ai_summary),
#             'originalShortDescription': safe_value(short_desc),
#             'originalDescription': safe_value(desc),
#             'severityScore': safe_value(severity_score),
#             'frequencyScore': safe_value(frequency_score),
#             'impactScore': safe_value(impact_score),
#             'severityScoreNorm': round(severity_score / 10, 2),
#             'frequencyScoreNorm': round(frequency_score / 20, 2),
#             'impactScoreNorm': round(impact_score / 30, 2),
#             'riskLevel': risk_level,
#             'solution': safe_value(ai_suggestion or '無提供解法'),
#             'location': safe_value(row.get('Location')),
#             'analysisTime': analysis_time,
#             'weights': {k: round(v / 10, 2) for k, v in weights.items()},
#         }

#     except Exception as e:
#         print(f"❌ 分析第 {idx+1} 筆失敗：", e)
#         return None











# def analyze_excel(filepath, weights=None):
#         # 預設權重設定（可被覆蓋）
#     default_weights = {
#         'keyword': 5.0,
#         'multi_user': 3.0,
#         'escalation': 2.0,
#         'config_item': 5.0,
#         'role_component': 3.0,
#         'time_cluster': 2.0
#     }
#     weights = {**default_weights, **(weights or {})}  # 合併預設權重與使用者提供的權重設定
#     print("🎛️ 使用中的權重設定：", weights)
#     print("🔍 開始分析 Excel 檔案...")
#     print(f"\n📂 讀取 Excel：{filepath}")
#     df = pd.read_excel(filepath)  # 讀取 Excel 檔案
#     print(f"📊 共讀取 {len(df)} 筆資料\n")
#     component_counts = df['Role/Component'].value_counts()  # 計算每個角色/元件的出現次數
#     df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')  # 將 'Opened' 欄位轉為日期格式
#     results = []  # 儲存分析結果
#     configuration_item_counts = df['Configuration item'].value_counts()  # 計算每個配置項的出現次數
#     configuration_item_max = configuration_item_counts.max()  # 找出配置項的最大出現次數
#     analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print(f"📅 分析時間：{analysis_time}")

#     for idx, row in tqdm(df.iterrows(), total=len(df), desc="📊 分析進度"):
#         print(f"\n🔍 第 {idx + 1} 筆分析中...")
#         description_text = row.get('Description', 'not filled')  # 取得描述文字
#         short_description_text = row.get('Short description', 'not filled') # 取得簡短描述文字
#         close_note_text = row.get('Close notes', 'not filled')  # 取得關閉註解文字
#         print(f"📄 描述：{description_text}")
#         print(f"🔑 簡短描述：{short_description_text}")
#         print(f"🔒 關閉註解：{close_note_text}")
#         # 這裡可以加入對描述文字的預處理，例如去除多餘空格、轉為小寫等
#         # description_text = normalize_text(description_text)  # 標準化文字    


#         #這裡要改成使用語意分析模型

#         keyword_score = is_high_risk(short_description_text)  # 計算關鍵字分數
#         print(f"⚠️ 高風險語意分數（keyword_score）：{keyword_score}")
#         user_impact_score = is_multi_user(description_text)  # 計算使用者影響分數
#         print(f"👥 多人影響分數（user_impact_score）：{user_impact_score}")
#         escalation_score = is_escalated(close_note_text)  # 計算升級處理分數
#         print(f"📈 升級處理分數（escalation_score）：{escalation_score}")



#         config_raw = configuration_item_counts.get(row.get('Configuration item'), 0)  # 取得配置項的出現次數
#         configuration_item_freq = config_raw / configuration_item_max if configuration_item_max > 0 else 0  # 計算配置項頻率

#         role_comp = row.get('Role/Component', 'not filled')  # 取得角色/元件
#         count = component_counts.get(role_comp, 0)  # 取得角色/元件的出現次數
#         if count >= 5:
#             role_component_freq = 3
#         elif count >= 3:
#             role_component_freq = 2
#         elif count == 2:
#             role_component_freq = 1
#         else:
#             role_component_freq = 0

#         this_time = row.get('Opened', 'not filled')  # 取得開啟時間
#         if pd.isnull(this_time):  # 如果開啟時間為空
#             time_cluster_score = 1
#         else:
#             others = df[df['Role/Component'] == role_comp]  # 篩選相同角色/元件的資料
#             close_events = others[(others['Opened'] >= this_time - pd.Timedelta(hours=24)) &
#                                   (others['Opened'] <= this_time + pd.Timedelta(hours=24))]  # 找出 24 小時內的事件
#             count_cluster = len(close_events)  # 計算事件數量
#             if count_cluster >= 3:
#                 time_cluster_score = 3
#             elif count_cluster == 2:
#                 time_cluster_score = 2
#             else:
#                 time_cluster_score = 1

#         severity_score = round(
#             keyword_score * weights['keyword'] +
#             user_impact_score * weights['multi_user'] +
#             escalation_score * weights['escalation'], 2
#         )

#         frequency_score = round(
#             configuration_item_freq * weights['config_item'] +
#             role_component_freq * weights['role_component'] +
#             time_cluster_score * weights['time_cluster'], 2
#         )


        
#         print(f"📊 嚴重性分數：{severity_score}，頻率分數：{frequency_score}")
#         print("🧠 頻率分數細項：")
#         print(f"🔸 配置項（Configuration Item）出現比例：{configuration_item_freq:.2f}，乘以權重後得 {configuration_item_freq * weights['config_item']:.2f} 分")
#         print(f"🔸 元件或角色（Role/Component）在整體中出現 {count} 次 → 給 {role_component_freq * weights['role_component']:.2f} 分")
#         print(f"🔸 在 24 小時內有 {count_cluster} 筆同元件事件 → 群聚加分 {time_cluster_score * weights['time_cluster']:.2f} 分")
#         print(f"📊 頻率總分 = {frequency_score}\n")




#         # 計算影響分數
#         impact_score = round(math.sqrt(severity_score**2 + frequency_score**2), 2)
#         risk_level = get_risk_level(impact_score)
#         print(f"📉 嚴重性：{severity_score}, 頻率：{frequency_score}, 總分(After KMean process)：{impact_score} → 分級：{risk_level}")
#         desc = str(row.get('Description', "")).strip()
#         short_desc = str(row.get('Short Description', "")).strip()
#         close_notes = str(row.get('Close notes', "")).strip()
#         resolution_text = f"{desc}\n{short_desc}\n{close_notes}".strip()

#         print(f"📦 Resolution 原始文字：{resolution_text}")  # ✅ 確認原始欄位內容

#         ai_suggestion = extract_resolution_suggestion(resolution_text)
#         print(f"🤖 GPT 建議句回傳：{ai_suggestion}")  # ✅ 確認 GPT 是否成功回應

#         # ✅ 安全地建立 AI 摘要輸入（若全空則顯示無資料）
#         summary_input_text = f"{short_desc}\n{desc}".strip()
#         if not summary_input_text:
#             summary_input_text = "（無原始摘要輸入）"

#         # ✅ 呼叫 GPT 摘要函式
#         ai_summary = extract_problem_with_custom_prompt(summary_input_text)

#         print(f"📦 Resolution 原始文字：{resolution_text}")
#         print(f"📝 AI 摘要輸入：{summary_input_text}")
#         print(f"🤖 GPT 摘要回傳：{ai_summary}")

#         # 儲存分析結果
#         results.append({
#             'id': safe_value(row.get('Incident') or row.get('Number')),
#             'configurationItem': safe_value(row.get('Configuration item')),
#             'roleComponent': safe_value(row.get('Role/Component')),
#             'subcategory': safe_value(row.get('Subcategory')),
#             'aiSummary': safe_value(ai_summary),
#             'originalShortDescription': safe_value(short_desc),
#             'originalDescription': safe_value(desc),
#             'severityScore': safe_value(severity_score),
#             'frequencyScore': safe_value(frequency_score),
#             'impactScore': safe_value(impact_score),
#             'severityScoreNorm': round(severity_score / 10, 2),
#             'frequencyScoreNorm': round(frequency_score / 20, 2),
#             'impactScoreNorm': round(impact_score / 30, 2),
#             'riskLevel': safe_value(get_risk_level(impact_score)),
#             'solution': safe_value(ai_suggestion or '無提供解法'),
#             'location': safe_value(row.get('Location')),
#             'analysisTime': analysis_time,
#             'weights': {k: round(v / 10, 2) for k, v in weights.items()},
#         })

#         # solution_text = row.get('Close notes') or '無提供解法'
#         recommended = recommend_solution(short_description_text)
#         keywords = extract_keywords(short_description_text)

#         print(f"✅ 已儲存 solution：{results[-1]['solution']}")
#         print(f"💡 建議解法：{recommended}")
#         print(f"🔑 抽取關鍵字：{keywords}")
#         print("—" * 250)  # 分隔線




#     # ⬇⬇⬇ KMeans 分群邏輯（支援三條件） ⬇⬇⬇
#     all_scores = [r['impactScore'] for r in results]
#     score_range = max(all_scores) - min(all_scores)
#     score_std = np.std(all_scores)

#     print(f"📈 分群判斷指標：count={len(all_scores)}, range={score_range:.2f}, stddev={score_std:.2f}")

#     if (
#         len(all_scores) >= KMEANS_MIN_COUNT and
#         score_range >= KMEANS_MIN_RANGE and
#         score_std >= KMEANS_MIN_STDDEV
#     ):
#         kmeans = KMeans(n_clusters=4, random_state=42)
#         labels = kmeans.fit_predict(np.array(all_scores).reshape(-1, 1))
#         centroids = kmeans.cluster_centers_.flatten()
#         set_kmeans_thresholds_from_centroids(centroids)
#         print(f"📊 KMeans 分群標籤：{labels}")
#         label_map = {}
#         for i, idx in enumerate(np.argsort(centroids)[::-1]):
#             label_map[idx] = ['高風險', '中風險', '低風險', '忽略'][i]
#         for i, r in enumerate(results):
#             r['riskLevel'] = label_map[labels[i]]
#         print(f"📌 KMeans 分群中心：{sorted(centroids, reverse=True)}")
#     else:
#         print("⚠️ 不啟用 KMeans，改用固定門檻分級")
#         for r in results:
#             r['riskLevel'] = get_risk_level(r['impactScore'])
#     # ⬆⬆⬆ 分群邏輯結束 ⬆⬆⬆
#     print("\n✅ 所有資料分析完成！")
#     return {
#         'data': results,
#         'analysisTime': analysis_time
#     }








