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
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg, model, history: chatHistory })
    });
    const data = await res.json();
    typingIndicator.remove();

    const reply = data.reply || data.error || "⚠️ 回應失敗";
    chatHistory.push({ role: "assistant", content: reply });

    const botMsg = document.createElement("div");
    botMsg.className = "msg bot";
    botMsg.innerHTML = `🤖 ${renderMessage(reply)}<span class="timestamp">${timestamp}</span>`;
    box.appendChild(botMsg);
scrollToBottom();

    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
  } catch (err) {
    typingIndicator.remove();
    const errorMsg = document.createElement("div");
    errorMsg.className = "msg bot";
    errorMsg.textContent = `⚠️ 錯誤：${err.message}`;
    box.appendChild(errorMsg);
  }
});

document.getElementById("showHistoryBtn").addEventListener("click", async () => {
  const listUI = document.getElementById("historyListUI");
  const wrapper = document.getElementById("chatHistoryList");
  const btn = document.getElementById("showHistoryBtn");

  const isVisible = wrapper.style.display !== "none";
  if (isVisible) {
    wrapper.style.display = "none";
    listUI.innerHTML = ""; // ✅ 收合時清空列表內容
    btn.innerText = "📂 展開完整歷史";
    return;
  } else {
    wrapper.style.display = "block";
    btn.innerText = "📂 收合歷史";
  }

  // 🔄 載入中...
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
      li.className = "list-group-item d-flex justify-content-between align-items-center";
      li.innerHTML = `
        📝 ${item.model} / ${new Date(item.timestamp).toLocaleString()}
        <button class="btn btn-sm btn-outline-primary">載入</button>
      `;

      li.querySelector("button").onclick = async () => {
        const res = await fetch(`/chat-history/${item.id}`);
        const json = await res.json();
        chatHistory = json.history || [];
        localStorage.setItem("chatHistory", JSON.stringify(chatHistory));

        // ✅ 標記 active
        document.querySelectorAll('#historyListUI .list-group-item').forEach(el => el.classList.remove('active'));
        li.classList.add('active');

        // ✅ 顯示訊息
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
      listUI.appendChild(li);
    });
  } catch (err) {
    listUI.innerHTML = `<li class="list-group-item text-danger">❌ 載入失敗</li>`;
    console.error(err);
  }

  
});

// ✅ 頁面一載入就自動展開歷史並載入清單
window.addEventListener("DOMContentLoaded", () => {
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

