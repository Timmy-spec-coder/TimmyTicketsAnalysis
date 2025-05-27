function navigateTo1(page) {
  const paths = {
    upload: "/",
    result: "/result",
    history: "/history",
    cluster: "/generate_cluster",
    manual: "/manual_input",
    gpt_prompt: "/gpt_prompt",
    chat: "/chat_ui"
  };
  window.location.href = paths[page] || "/";
}

function renderMessage(content) {
  return content
    .replace(/```([^`]+)```/gs, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, "<br>");
}


function scrollToBottom() {
  const box = document.getElementById("chatBox");
  box.scrollTo({ top: box.scrollHeight, behavior: "smooth" });
}

let chatHistory = JSON.parse(localStorage.getItem("chatHistory") || "[]");






window.addEventListener("DOMContentLoaded", () => {

  const chatInput = document.getElementById("chatInput");

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault(); // 阻止預設換行
    document.getElementById("chatForm").requestSubmit(); // 模擬送出表單
  }
});

  // 自動高度調整
  chatInput.setAttribute("style", "height:auto;overflow-y:hidden;");
  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = chatInput.scrollHeight + "px";
  });

  const toggleBtn = document.getElementById("toggleDarkMode");
  const sidebarToggle = document.getElementById("sidebarToggle");

  const isDark = localStorage.getItem("dark-mode") === "true";
  document.body.classList.toggle("dark-mode", isDark);
  if (toggleBtn) toggleBtn.innerHTML = isDark ? "🌞 淺色模式" : "🌙 深色模式";

  const isCollapsed = localStorage.getItem("sidebarCollapsed") === "true";
  document.body.classList.toggle("sidebar-collapsed", isCollapsed);
  if (sidebarToggle) sidebarToggle.textContent = isCollapsed ? "→" : "←";

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const nowDark = document.body.classList.toggle("dark-mode");
      localStorage.setItem("dark-mode", nowDark);
      toggleBtn.innerHTML = nowDark ? "🌞 淺色模式" : "🌙 深色模式";
    });
  }



  // 側邊欄切換
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      const collapsed = document.body.classList.toggle("sidebar-collapsed");
      localStorage.setItem("sidebarCollapsed", collapsed);
      sidebarToggle.textContent = collapsed ? "→" : "←";
    });
  }

  // 🧠 若有歷史紀錄，自動載入畫面
  const box = document.getElementById("chatBox");
  chatHistory.forEach(entry => {
    const div = document.createElement("div");
    div.className = "msg " + (entry.role === "user" ? "user" : "bot");
    div.innerHTML = `${entry.role === "user" ? "👤" : "🤖"} ${renderMessage(entry.content)}`;
    box.appendChild(div);
  });
  scrollToBottom();
});




document.getElementById("chatForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = document.getElementById("chatInput");
  const msg = input.value.trim();
  if (!msg) return;
    // 🛡️ 若建置中，直接擋掉提問
  if (window.kbBuilding) {
    const box = document.getElementById("chatBox");
    const div = document.createElement("div");
    div.className = "msg bot";
    div.innerHTML = `🤖 資料庫正在建置中，暫時無法回答問題，請稍後再試。`;
    box.appendChild(div);
    scrollToBottom();
    return;
  }


  const box = document.getElementById("chatBox");
  const timestamp = new Date().toLocaleTimeString();
  const model = document.getElementById("modelSelect")?.value || "mistral";

  const userMsg = document.createElement("div");
  userMsg.className = "msg user";
  userMsg.innerHTML = `👤 ${msg}<span class="timestamp">${timestamp}</span>`;
  box.appendChild(userMsg);
  input.value = "";

  const typingIndicator = document.createElement("div");
  typingIndicator.className = "msg bot";
  typingIndicator.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
  box.appendChild(typingIndicator);
scrollToBottom();

  chatHistory.push({ role: "user", content: msg });

  try {


    const chatId = localStorage.getItem("currentChatId");

    const payload = {
    message: msg,
    model,
    history: chatHistory,
    chatId: chatId
  };

  console.log("[🚀 發送訊息 Payload]", payload); // ✅ 新增這行

    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        model,
        history: chatHistory,
        chatId: chatId
      })
    });




    const data = await res.json();
    typingIndicator.remove();

    const reply = data.reply || data.error || "⚠️ 回應失敗";






    const botMsg = document.createElement("div");
    botMsg.className = "msg bot";
    botMsg.innerHTML = `🤖 <span id="typingContent"></span><span class="timestamp">${timestamp}</span>`;
    box.appendChild(botMsg);

    const contentSpan = botMsg.querySelector("#typingContent");
    const finalReply = renderMessage(reply);  // 支援換行與程式碼區塊等
    const finalChars = Array.from(finalReply); // 防止破壞 HTML 結構（支援 emoji、中文、特殊字）

    let i = 0;


let tempReply = "";  // 暫存純文字

const typingInterval = setInterval(() => {
  if (i >= finalChars.length) {
    clearInterval(typingInterval);

    // ✅ 打字完畢才套用 renderMessage() 美化
    contentSpan.innerHTML = renderMessage(tempReply);

    const assistantReply = { role: "assistant", content: reply };
    chatHistory.push(assistantReply);
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

    scrollToBottom();
    return;
  }

  tempReply += finalChars[i++];
  contentSpan.textContent = tempReply;  // ⚠️ 逐字用 textContent 保留純文字
  scrollToBottom();
}, 20);




    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

// ✅ 強制刷新側邊欄（如果歷史區塊是開的）
const showBtn = document.getElementById("showHistoryBtn");
const historyWrapper = document.getElementById("chatHistoryList");

if (showBtn && historyWrapper && historyWrapper.style.display !== "none") {
  showBtn.click(); // 關閉
  setTimeout(() => showBtn.click(), 200); // 展開刷新
}


  } catch (err) {
    typingIndicator.remove();
    const errorMsg = document.createElement("div");
    errorMsg.className = "msg bot";
    errorMsg.textContent = `⚠️ 錯誤：${err.message}`;
    box.appendChild(errorMsg);
  }
});





// 📂 展開 / 收合歷史紀錄
document.getElementById("showHistoryBtn").addEventListener("click", async () => {
  const listUI = document.getElementById("historyListUI");
  const wrapper = document.getElementById("chatHistoryList");
  const btn = document.getElementById("showHistoryBtn");

  const isVisible = wrapper.style.display !== "none";
  if (isVisible) {
    wrapper.style.display = "none";
    listUI.innerHTML = "";
    btn.innerText = "📂 展開完整歷史";
    return;
  } else {
    wrapper.style.display = "block";
    btn.innerText = "📂 收合歷史";
  }

  listUI.innerHTML = `<li class="list-group-item">🔄 載入中...</li>`;

  try {
    const res = await fetch("/chat-history-list");
    const data = await res.json();

    if (!data || data.length === 0) {
      listUI.innerHTML = `<li class="list-group-item text-muted">📭 尚無歷史紀錄</li>`;
      return;
    }

    listUI.innerHTML = "";

    data.forEach(item => {
      const li = document.createElement("li");
      li.className = "list-group-item";

      li.innerHTML = `
        <div class="d-flex justify-content-between align-items-start">
          <div class="history-click-target" style="cursor: pointer;">
            📝 <strong>${item.title || "（無標題）"}</strong><br>
            <small class="text-muted">${new Date(item.timestamp).toLocaleString()} ・${item.model}</small>
          </div>
          <div class="dropdown">
            <button class="btn btn-sm btn-link text-muted dropdown-toggle dropdown-toggle-no-caret" type="button" data-bs-toggle="dropdown" aria-expanded="false">
              ⋯
            </button>
            <ul class="dropdown-menu dropdown-menu-end">
              <li><a class="dropdown-item rename-btn" href="#">✏️ 編輯標題</a></li>
              <li><a class="dropdown-item text-danger delete-btn" href="#">🗑️ 刪除話題</a></li>
            </ul>
          </div>
        </div>
      `;

      // ✅ 點整塊標題區塊就載入對話
      li.querySelector(".history-click-target").onclick = async () => {
        const res = await fetch(`/chat-history/${item.id}`);
        const json = await res.json();

        chatHistory = json.history || [];
        localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
        localStorage.setItem("currentChatId", json.id);

        document.querySelectorAll('#historyListUI .list-group-item').forEach(el => el.classList.remove('active'));
        li.classList.add('active');

        const box = document.getElementById("chatBox");
        box.innerHTML = "";
        chatHistory.forEach(entry => {
          const div = document.createElement("div");
          div.className = "msg " + (entry.role === "user" ? "user" : "bot");
          div.innerHTML = `${entry.role === "user" ? "👤" : "🤖"} ${renderMessage(entry.content)}`;
          box.appendChild(div);
        });
        scrollToBottom();
      };

      // ✏️ 編輯標題
      li.querySelector(".rename-btn").onclick = async () => {
        const newTitle = prompt("請輸入新的話題名稱：", item.title || "");
        if (!newTitle) return;

        try {
          const res = await fetch("/rename-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ chatId: item.id, newTitle })
          });
          const result = await res.json();
          if (result.success) {
            alert("✅ 標題已更新！");
            document.getElementById("showHistoryBtn").click();
            setTimeout(() => document.getElementById("showHistoryBtn").click(), 200);
          } else {
            alert("❌ 無法更新標題：" + (result.error || ""));
          }
        } catch (err) {
          alert("❌ 發送錯誤：" + err.message);
        }
      };

      // 🗑️ 刪除話題
      li.querySelector(".delete-btn").onclick = async () => {
        if (!confirm(`是否刪除這個話題？\n「${item.title || item.id}」`)) return;

        try {
          const res = await fetch(`/delete-chat/${item.id}`, { method: "DELETE" });
          const result = await res.json();
          if (result.success) {
            alert("🗑️ 話題已刪除！");
            document.getElementById("showHistoryBtn").click();
            setTimeout(() => document.getElementById("showHistoryBtn").click(), 200);
          } else {
            alert("❌ 刪除失敗：" + (result.error || ""));
          }
        } catch (err) {
          alert("❌ 發送錯誤：" + err.message);
        }
      };

      listUI.appendChild(li);

      // ✅ 初始化 dropdown
      const dropdownBtn = li.querySelector(".dropdown-toggle");
      if (dropdownBtn) new bootstrap.Dropdown(dropdownBtn);
    });

    // ✅ 搜尋篩選
    document.getElementById("historySearchInput").addEventListener("input", e => {
      const keyword = e.target.value.trim().toLowerCase();
      const items = document.querySelectorAll("#historyListUI li");

      items.forEach(li => {
        const text = li.textContent.toLowerCase();
        li.style.display = text.includes(keyword) ? "" : "none";
      });
    });

  } catch (err) {
    listUI.innerHTML = `<li class="list-group-item text-danger">❌ 載入失敗</li>`;
    console.error("[載入歷史錯誤]", err);
  }
});


// 📊 顯示統計資訊
function sendStat(field) {
  const input = document.getElementById("chatInput");
  input.value = `Please show statistics of ${field}`;
  document.getElementById("chatForm").requestSubmit();
}



// ✅ 頁面一載入就自動展開歷史並載入清單
window.addEventListener("DOMContentLoaded", () => {
  createNewChatSession(); // ✅ 每次進入頁面時都建立新話題
  document.getElementById("showHistoryBtn").click();
});

// 🧹 清空聊天
document.getElementById("clearChatBtn").addEventListener("click", () => {
  if (confirm("你確定要清空當前聊天紀錄？此操作無法復原")) {
    localStorage.removeItem("chatHistory");
    chatHistory = [];
    const box = document.getElementById("chatBox");
    box.innerHTML = "";
    scrollToBottom(); // ✅ 滾到底部，避免有滾動殘留
  }
});



function createNewChatSession() {
  const now = new Date();
  const random = Math.random().toString(36).substring(2, 6);  // 生成 4 位小寫亂數
  const id = now.toISOString().slice(0, 16).replace(/[-:T]/g, "") + "_" + random + "_chat";


  const model = document.getElementById("modelSelect")?.value || "mistral";

  // 儲存到 localStorage
  localStorage.setItem("currentChatId", id);
  localStorage.setItem("chatHistory", JSON.stringify([]));

  // 清空畫面
  const box = document.getElementById("chatBox");
  box.innerHTML = "";

  // 清空 UI 歷史記錄
  const listUI = document.getElementById("historyListUI");
  if (listUI) listUI.querySelectorAll(".list-group-item").forEach(el => el.classList.remove("active"));

  // 可加入一個小提示
  const welcome = document.createElement("div");
  welcome.className = "msg bot";
  welcome.innerHTML = `🤖 <em>已建立新話題，可開始提問！</em>`;
  box.appendChild(welcome);
}


