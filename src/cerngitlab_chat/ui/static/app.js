(() => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const sendBtn = document.getElementById("send-btn");
  const transcript = document.getElementById("transcript");
  const emptyState = document.getElementById("empty-state");
  const modelLabel = document.getElementById("model-label");
  const statusLabel = document.getElementById("status-label");
  const clearBtn = document.getElementById("clear-btn");

  function hideEmpty() {
    if (emptyState) emptyState.remove();
    const existing = document.getElementById("empty-state");
    if (existing) existing.remove();
  }

  function appendBubble(role, text, isError = false) {
    hideEmpty();
    const bubble = document.createElement("article");
    bubble.className = `bubble ${isError ? "error" : role}`;
    const roleEl = document.createElement("span");
    roleEl.className = "role";
    roleEl.textContent = isError ? "error" : role;
    const body = document.createElement("div");
    body.textContent = text;
    bubble.append(roleEl, body);
    transcript.appendChild(bubble);
    transcript.scrollTop = transcript.scrollHeight;
  }

  function appendToolLogs(toolLogs, usedTools) {
    hideEmpty();
    const details = document.createElement("details");
    details.className = "tool-log";
    details.open = true;

    const summary = document.createElement("summary");
    const calls = (toolLogs || []).filter((e) => e.kind === "call").length;
    summary.textContent = usedTools
      ? `MCP tool log (${calls} call${calls === 1 ? "" : "s"})`
      : "MCP tool log (no tools used)";
    details.appendChild(summary);

    if (!toolLogs || toolLogs.length === 0) {
      const warn = document.createElement("div");
      warn.className = "warn";
      warn.textContent =
        "No MCP tools were called this turn. Treat the answer as unverified.";
      details.appendChild(warn);
    } else {
      const list = document.createElement("ol");
      for (const entry of toolLogs) {
        const li = document.createElement("li");
        const dir = document.createElement("span");
        dir.className = "dir";
        dir.textContent = entry.kind === "call" ? "→ call " : "← result ";
        li.appendChild(dir);
        li.appendChild(document.createTextNode(entry.tool));
        if (entry.detail) {
          li.appendChild(document.createTextNode(` ${entry.detail}`));
        }
        list.appendChild(li);
      }
      details.appendChild(list);
    }

    transcript.appendChild(details);
    transcript.scrollTop = transcript.scrollHeight;
  }

  async function refreshStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      modelLabel.textContent = data.model || "model";
      statusLabel.textContent = data.ready ? "ready" : "starting";
    } catch {
      statusLabel.textContent = "offline";
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    appendBubble("user", message);
    input.value = "";
    sendBtn.disabled = true;
    statusLabel.textContent = "thinking";

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || `Request failed (${res.status})`);
      }
      appendToolLogs(data.tool_logs || [], Boolean(data.used_tools));
      appendBubble("assistant", data.reply || "");
      if (data.model) modelLabel.textContent = data.model;
      statusLabel.textContent = data.used_tools ? "ready" : "no tools";
    } catch (err) {
      appendBubble("assistant", err.message || String(err), true);
      statusLabel.textContent = "error";
    } finally {
      sendBtn.disabled = false;
      input.focus();
    }
  });

  clearBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/clear", { method: "POST" });
    } catch {
      /* ignore network blip; still clear the UI */
    }
    transcript.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.id = "empty-state";
    empty.textContent =
      "Try questions like “Where is ParticleAdder used in Moore?” or “What projects make up the LHCb software stack?”";
    transcript.appendChild(empty);
    statusLabel.textContent = "ready";
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  refreshStatus();
  setInterval(refreshStatus, 15000);
  input.focus();
})();
