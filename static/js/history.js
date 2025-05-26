// 當 DOM 完全加載後執行




window.addEventListener('DOMContentLoaded', () => {
    // 獲取顯示歷史紀錄的列表元素
    const historyList = document.getElementById('historyList');
    // 從 localStorage 取得歷史紀錄資料，若無資料則設為空陣列
    const savedHistory = JSON.parse(localStorage.getItem('historyData') || '[]');
    const noHistoryMsg = document.getElementById('no-history-msg');
    if (savedHistory.length === 0) {
        noHistoryMsg.style.display = 'block';
    } else {
        noHistoryMsg.style.display = 'none';
    }


    // 清空原本的內容
    historyList.innerHTML = '';

    // 遍歷歷史紀錄資料，將每一項添加到列表中
    savedHistory.forEach(item => {
        const li = document.createElement('div');
        li.className = "col-12 col-sm-6 col-md-4"; // ⭐⭐ 加上這個很關鍵！

        li.innerHTML = `
            <div class="history-item card p-3 h-100 shadow-sm">
                <div class="history-header d-flex justify-content-between align-items-center">
                    <h5 class="mb-1">${item.file}</h5>
                    <small class="text-extra-muted">${item.time}</small>

                </div>
                <p class="mb-2 text-secondary">${item.summary}</p>
                <div class="btn-group mt-auto" role="group">
                    <a href="/get-json?file=${item.uid}.json" target="_blank"
                    class="btn btn-sm btn-outline-info">🧾 預覽 JSON</a>
                    <a href="/download-excel?uid=${item.uid}" download
                    class="btn btn-sm btn-outline-success">📥 分析 Excel</a>
                    <a href="/download-original?uid=${item.uid}" download
                    class="btn btn-sm btn-outline-secondary">📤 原始 Excel</a>
                </div>
            </div>
        `;
        historyList.appendChild(li);
    });





    // 初始化深色模式
    const isDark = localStorage.getItem('dark-mode') === 'true'; // 從 localStorage 取得深色模式狀態
    if (isDark) {
        // 如果是深色模式，添加深色模式的樣式
        document.body.classList.add('dark-mode');
        // 更新深色模式按鈕的文字
        document.getElementById('toggleDarkMode').innerHTML = '🌞 淺色模式';
    } else {
        // 如果不是深色模式，移除深色模式的樣式
        document.body.classList.remove('dark-mode');
        // 更新深色模式按鈕的文字
        document.getElementById('toggleDarkMode').innerHTML = '🌙 深色模式';
    }

    // 獲取側邊欄切換按鈕
    const toggleBtn = document.getElementById('sidebarToggle');
    if (toggleBtn) {
        // 根據側邊欄的狀態更新按鈕文字
        toggleBtn.textContent = document.body.classList.contains('sidebar-collapsed') ? '→' : '←';
    }
});

// 為深色模式切換按鈕添加點擊事件
document.getElementById('toggleDarkMode').addEventListener('click', () => {
    // 切換深色模式的樣式
    document.body.classList.toggle('dark-mode');
    // 獲取當前是否為深色模式
    const isDark = document.body.classList.contains('dark-mode');
    // 將深色模式狀態存入 localStorage
    localStorage.setItem('dark-mode', isDark);
    // 根據模式更新按鈕文字
    document.getElementById('toggleDarkMode').innerHTML = isDark ? '🌞 淺色模式' : '🌙 深色模式';
});

// 定義函數：切換側邊欄的顯示狀態
function toggleSidebar() {
    // 切換側邊欄的樣式
    document.body.classList.toggle('sidebar-collapsed');
    // 獲取側邊欄切換按鈕
    const toggleBtn = document.getElementById('sidebarToggle');
    // 根據側邊欄的狀態更新按鈕文字
    toggleBtn.textContent = document.body.classList.contains('sidebar-collapsed') ? '→' : '←';
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

// 為清除歷史紀錄按鈕添加點擊事件
document.getElementById('clearHistoryBtn').addEventListener('click', () => {
    // 確認是否清除歷史紀錄
    if (confirm('你確定要清除所有歷史紀錄嗎？這個操作無法復原。')) {
        // 從 localStorage 移除歷史紀錄資料
        localStorage.removeItem('historyData');
        localStorage.removeItem('historyHTML');
        // 清空歷史紀錄列表的內容
        document.getElementById('historyList').innerHTML = '';
        noHistoryMsg.style.display = 'block';
        if (savedHistory.length === 0) {
        noHistoryMsg.style.display = 'block';
        setTimeout(() => noHistoryMsg.classList.add('show'), 10);
        } 
        else {
        noHistoryMsg.classList.remove('show');
        setTimeout(() => noHistoryMsg.style.display = 'none', 300);
}


        // 顯示清除成功的提示訊息
        alert('✅ 歷史紀錄已清除！');
    }
});


function addHistoryItem(uid, fileName, summaryText) {
    const now = new Date();
    const time = now.toLocaleTimeString();
    const record = {
        uid,
        file: fileName,
        time,
        summary: summaryText
    };

    const stored = JSON.parse(localStorage.getItem('historyData') || '[]');
    stored.unshift(record);
    localStorage.setItem('historyData', JSON.stringify(stored));

    const li = document.createElement('li');
    li.innerHTML = `
        <strong>${fileName}</strong> - ${time}<br>
        <span>${summaryText}</span><br>
        <a href="/get-json?file=${uid}.json" target="_blank">🧾 預覽 JSON</a> |
        <a href="/download-excel?uid=${uid}" download>📥 分析 Excel</a> |
        <a href="/download-original?uid=${uid}" download>📤 原始 Excel</a>
    `;
    historyList.prepend(li);
    console.log("📦 加入歷史記錄，UID =", uid);
    console.log("📦 檔案名稱 =", fileName);
}
