window.addEventListener('DOMContentLoaded', async () => {
    // ===== 深色模式初始化 =====
    const isDark = localStorage.getItem('dark-mode');
    const toggleBtn = document.getElementById('toggleDarkMode');
    const sidebarToggle = document.getElementById('sidebarToggle');

    if (isDark === null || isDark === 'true') {
        document.body.classList.add('dark-mode');
        if (toggleBtn) toggleBtn.innerHTML = '🌞 淺色模式';
    } else {
        document.body.classList.remove('dark-mode');
        if (toggleBtn) toggleBtn.innerHTML = '🌙 深色模式';
    }

    if (sidebarToggle) {
        sidebarToggle.textContent = document.body.classList.contains('sidebar-collapsed') ? '→' : '←';
    }

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const isDark = document.body.classList.contains('dark-mode');
            toggleBtn.innerHTML = isDark ? '🌞 淺色模式' : '🌙 深色模式';
            localStorage.setItem('dark-mode', isDark);
                // 🌈 切換模式後重新渲染雷達圖
            if (window.renderAllCharts) window.renderAllCharts();
        });
    }

    // ===== 動態載入分析結果卡片 =====
    const container = document.getElementById('resultCards');
    const riskLevelToClass = (level) => {
        switch(level) {
            case '高風險': return 'risk-critical';
            case '中風險': return 'risk-high';
            case '低風險': return 'risk-medium';
            case '忽略': default: return 'risk-low';
        }
    };
    if (!container) return;
    try {
        document.getElementById('filterLoading').style.display = 'flex';
        container.innerHTML = ''; // 清除原卡片


        const res = await fetch('/get-results');       // ✅ 補上這行
        const resultJson = await res.json();           // ✅ 正確解析 JSON
        const data = resultJson.data;
const weights = resultJson.weights || {};  // ✅ 這行要先定義 weights

        
        console.log("📦 當次分析使用的權重設定：", weights);





        const filterRange = document.getElementById('filterRange');
        let rangeDays = localStorage.getItem('filter-days');
        if (rangeDays === null) rangeDays = '7'; // 預設值
        if (filterRange) {
            filterRange.value = rangeDays;
            filterRange.addEventListener('change', () => {
                const val = filterRange.value;
                    if (val === 'all') {
                        localStorage.setItem('filter-days', val);
                    } 
                    else {
                        localStorage.setItem('filter-days', val);}
                    location.reload();
                });
            }

            let filterStartDate = null;
            if (rangeDays !== 'all') {
                const now = new Date();
                if (rangeDays === '0') {
                        // 只顯示今天（00:00 起）
                    filterStartDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                    } else {
                        const days = parseInt(rangeDays);
                        if (!isNaN(days)) {
                            filterStartDate = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
                        }
                    }
                }

                if (!data || data.length === 0) {
                    container.innerHTML = '<p>⚠️ 尚無分析資料，請先回首頁上傳 Excel。</p>';
                    return;
                    }
                data.forEach(row => {
                    console.log("📌 原始 row.id：", row.id);
                    console.log("📌 row.weights：", row.weights);
                    console.log("📌 resultJson.weights（預設值）：", weights);

                            if (!row.analysisTime || isNaN(Date.parse(row.analysisTime))) return;
                            const rowDate = new Date(row.analysisTime);
                            if (filterStartDate && rowDate < filterStartDate) return;

const weightObj = row.weights ?? {}; // 拿掉 fallback，因為外層沒有了

console.log("📎 使用中的 weightObj：", weightObj);


                           const analysisWeights = resultJson.weights; // 👈 這就是「這次上傳的權重」
                            console.log("🧪 當筆資料 ID：", row.id, "使用權重：", row.weights);

                            const severityRaw = row.severityScore;
                            const frequencyRaw = row.frequencyScore;
                            const impactRaw = row.impactScore;

                            // ✅ 正規化：除以最大分數，保留兩位小數
                            const severityNorm = (severityRaw / 10).toFixed(2);
                            const frequencyNorm = (frequencyRaw / 20).toFixed(2);
                            const impactNorm = (impactRaw / 30).toFixed(2);


                            const cardRow = document.createElement('div');
                            cardRow.className = 'card-row';

                            const infoCard = document.createElement('div');
                            infoCard.className = 'card card-info';
infoCard.innerHTML = `
    <h3>🎯 Incident: ${row.id}</h3>
    <div class="card-grid">

<!-- Config Item -->
<div class="progress-block">
  <strong>Config Item:</strong>
  <span class="score-value">${row.configurationItem || '—'}</span>
</div>

<!-- Severity 正規化 -->
<div class="progress-block">
  <strong>Severity <span class="score-max">(0–1)</span>:</strong>
  <span class="score-value">${severityNorm}</span>
  <div class="progress-wrapper">
    <progress class="progress-bar" value="${severityNorm}" max="1" data-type="severity"></progress>
    <span class="progress-percent">${(severityNorm * 100).toFixed(0)}%</span>
  </div>
</div>

<!-- Frequency 正規化 -->
<div class="progress-block">
  <strong>Frequency <span class="score-max">(0–1)</span>:</strong>
  <span class="score-value">${frequencyNorm}</span>
  <div class="progress-wrapper">
    <progress class="progress-bar" value="${frequencyNorm}" max="1" data-type="frequency"></progress>
    <span class="progress-percent">${(frequencyNorm * 100).toFixed(0)}%</span>
  </div>
</div>

<!-- Impact 正規化 -->
<div class="progress-block">
  <strong>Impact <span class="score-max">(0–1)</span>:</strong>
  <span class="score-value">${impactNorm}</span>
  <div class="progress-wrapper">
    <progress class="progress-bar" value="${impactNorm}" max="1" data-type="impact"></progress>
    <span class="progress-percent">${(impactNorm * 100).toFixed(0)}%</span>
  </div>
</div>



        <div><strong>Issue Summary:</strong> <span>${row.aiSummary || '—'}</span></div> <!-- 👈 新增這行 -->
        <div><strong>Solution:</strong> <span>${row.solution || '—'}</span></div>
                <div><strong>Risk Level:</strong>
            <span class="badge ${riskLevelToClass(row.riskLevel)}">${row.riskLevel}</span></div>

<div style="display: flex; align-items: center; gap: 20px; margin-top: 6px;">
  <span style="background: #e3f2fd; color: #1565c0; border-radius: 7px; padding: 3px 12px; font-size: 1rem; display: flex; align-items: center;">
    <i class="fas fa-map-marker-alt" style="margin-right: 5px;"></i>
    ${row.location || '—'}
  </span>
  <span style="background: #fff8e1; color: #b26a00; border-radius: 7px; padding: 3px 12px; font-size: 1rem; display: flex; align-items: center;">
    <i class="fas fa-calendar-alt" style="margin-right: 5px;"></i>
    ${row.analysisTime || '—'}
  </span>
</div>








    </div>

<div class="weights-summary mt-3">
  <details>
    <summary>⚖️ 查看使用的權重設定</summary>
    <div class="weight-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px 16px; padding-top: 12px;">
        <div><strong>🔑 高風險語意：</strong> ${weightObj.keyword ?? '—'}</div>
        <div><strong>👥 多人受影響：</strong> ${weightObj.multi_user ?? '—'}</div>
        <div><strong>📈 升級處理：</strong> ${weightObj.escalation ?? '—'}</div>
        <div><strong>🧩 配置項頻率：</strong> ${weightObj.config_item ?? '—'}</div>
        <div><strong>🧑‍💻 元件角色頻率：</strong> ${weightObj.role_component ?? '—'}</div>
        <div><strong>⏱️ 群聚事件：</strong> ${weightObj.time_cluster ?? '—'}</div>
    </div>
  </details>
</div>






`;


                            const linker = document.createElement('div');
                            linker.className = 'card-linker';
                            linker.innerHTML = `<span>⇨</span>`;

                            // 📌 把圖表區塊包在外層容器中
                            const chartWrapper = document.createElement('div');
                            chartWrapper.className = 'card-chart-wrapper';
                            chartWrapper.innerHTML = `
                                <h4>視覺化分析</h4>
                            `;

                            const chartCard = document.createElement('div');
                            chartCard.className = 'card card-chart';
                            chartCard.innerHTML = `
                                <div class="card-visual-area"
                                    data-severity="${row.severityScore}" 
                                    data-frequency="${row.frequencyScore}" 
                                    data-impact="${row.impactScore}">
                                </div>
                            `;
                            chartWrapper.appendChild(chartCard);       // 🔁 圖表卡片加到 wrapper 裡
                            cardRow.appendChild(infoCard);
                            cardRow.appendChild(linker);
                            cardRow.appendChild(chartWrapper);         // 🟨 插入整個 wrapper
                            container.appendChild(cardRow);
                        });


Promise.resolve().then(() => {
    if (typeof window.renderAllCharts === 'function') {
        window.renderAllCharts();
    }

    document.querySelectorAll('.progress-wrapper').forEach(wrapper => {
        const bar = wrapper.querySelector('.progress-bar');
        const percentLabel = wrapper.querySelector('.progress-percent');

        const value = parseFloat(bar.getAttribute('value')) || 0;
        const max = parseFloat(bar.getAttribute('max')) || 100;

        // ✅ 重置
        bar.value = 0;
        percentLabel.textContent = `0%`;

        setTimeout(() => {
            // ✅ 先讓進度條填充到正確 value
            bar.value = value;

            const finalPercent = Math.round((value / max) * 100);

            // ✅ 這裡開始用 requestAnimationFrame 動態跑百分比數字
            let currentPercent = 0;
            const duration = 1000; // 1秒內跑完
            const startTime = performance.now();

            function animatePercent(time) {
                const elapsed = time - startTime;
                const progress = Math.min(elapsed / duration, 1); // progress 0~1
                currentPercent = Math.floor(finalPercent * progress);

                percentLabel.textContent = `${currentPercent}%`;

                if (progress < 1) {
                    requestAnimationFrame(animatePercent);
                } else {
                    percentLabel.textContent = `${finalPercent}%`; // 最後補精確
                }
            }

            requestAnimationFrame(animatePercent);

            // ✅ 設定進度條顏色
            let bg = '';
            let textColor = '';

            if (finalPercent < 35) {
                bg = 'linear-gradient(90deg, #6ee7b7, #3bceac)';
                textColor = '#4caf50';
            } else if (finalPercent < 70) {
                bg = 'linear-gradient(90deg, #ffe57f, #ffca28)';
                textColor = '#f9a825';
            } else {
                bg = 'linear-gradient(90deg, #ff8a80, #e53935)';
                textColor = '#e53935';
            }

            bar.style.setProperty('--progress-color', bg);
            percentLabel.style.color = textColor;

        }, 300); // 小延遲，讓動畫有呼吸感
    });
});


    } 
    catch (err) {
        console.error('🚨 無法取得結果：', err);
        container.innerHTML = '<p style="color:red;">❌ 無法載入分析結果。</p>';
        if (filterLoading) filterLoading.style.display = 'none'; // ✨補這行

    }

    document.getElementById('filterLoading').style.display = 'none';


const filterInput = document.getElementById('filterInput');
const clearBtn = document.getElementById('clearFilterBtn');
if (clearBtn) {
    clearBtn.addEventListener('click', () => {
        // 清除日期篩選器的值
        if (filterInput) filterInput.value = '';
        // 清除天數篩選器的值
        if (filterRange) filterRange.value = '7';
        // 清除 localStorage 中的天數篩選器值
        localStorage.removeItem('filter-days');
        location.reload();
    });
}

});


// 定義函數：切換側邊欄的顯示狀態
function toggleSidebar() {
    // 切換側邊欄的樣式
    document.body.classList.toggle('sidebar-collapsed');
    // 獲取側邊欄切換按鈕元素
    const toggleBtn = document.getElementById('sidebarToggle');
    // 根據側邊欄狀態更新按鈕文字
    if (document.body.classList.contains('sidebar-collapsed')) {
        toggleBtn.textContent = '→'; // 側邊欄收起時顯示箭頭向右
    } else {
        toggleBtn.textContent = '←'; // 側邊欄展開時顯示箭頭向左
    }
}

// 定義函數：平滑滾動到指定的元素
function navigateTo(id) {
    // 獲取目標元素
    const target = document.getElementById(id);
    if (target) {
        // 滾動到目標元素，並啟用平滑滾動效果
        target.scrollIntoView({ behavior: 'smooth' });
    }
}

// 定義函數：導航到指定的頁面
function navigateTo1(page) {
    if (page === 'upload') {
        // 導航到 Flask 的首頁路由
        window.location.href = '/';
    } else if (page === 'result') {
        // 導航到 Flask 的 /result 路由
        window.location.href = '/result';
    } else if (page === 'history') {
        // 導航到 Flask 的 /history 路由
        window.location.href = '/history';
    } else if (page === 'cluster') {
        window.location.href = '/generate_cluster';  // ✅ 對應後端路由名稱
    } else if (page === 'manual') {
        window.location.href = '/manual_input';  // ✅ 對應後端路由名稱
    } else if (page === 'gpt_prompt') {
        window.location.href = '/gpt_prompt';  // ✅ 對應後端路由名稱
    } else if (page === 'chat') {
        window.location.href = '/chat_ui';  // ✅ 對應後端路由名稱
    }
    
}