/**
 * MedSaathi frontend connects to the backend by creating a session (`POST /session/new`),
 * checking health (`GET /health`), then sending user queries to `POST /chat` or
 * `POST /chat/stream` and rendering responses in the chat timeline.
 */

const API_BASE = "http://localhost:8000";
const USE_STREAMING = true;

let sessionId = null;
let isWaitingForResponse = false;
let toastTimer = null;

const chatArea = document.getElementById("chat-area");
const welcomeCard = document.getElementById("welcome-card");
const typingIndicator = document.getElementById("typing-indicator");
const messageInput = document.getElementById("message-input");
const btnSend = document.getElementById("btn-send");
const btnNewChat = document.getElementById("btn-new-chat");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const crisisBanner = document.getElementById("crisis-banner");
const crisisClose = document.getElementById("crisis-close");
const errorToast = document.getElementById("error-toast");
const errorToastText = document.getElementById("error-toast-text");

/**
 * Bootstraps the page: wires listeners, creates an initial session, and checks API health.
 */
document.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  await initSession();
  await checkHealth();
});

/**
 * Registers all UI event listeners once.
 */
function bindEvents() {
  btnSend.addEventListener("click", () => {
    void sendMessage();
  });

  btnNewChat.addEventListener("click", () => {
    void newChat();
  });

  crisisClose.addEventListener("click", hideCrisisBanner);

  messageInput.addEventListener("input", autoResizeTextarea);

  messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query") || "";
      messageInput.value = query;
      autoResizeTextarea();
      void sendMessage();
    });
  });
}

/**
 * Creates a new backend chat session and stores `sessionId`.
 * @returns {Promise<void>}
 */
async function initSession() {
  try {
    const response = await fetch(`${API_BASE}/session/new`, { method: "POST" });
    if (!response.ok) {
      throw new Error(`Session init failed with ${response.status}`);
    }

    const data = await response.json();
    sessionId = data.session_id;
  } catch {
    showErrorToast("Server se connect nahi ho pa raha");
  }
}

/**
 * Checks backend readiness and updates the header connection indicator.
 * @returns {Promise<void>}
 */
async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) {
      throw new Error(`Health failed with ${response.status}`);
    }

    statusDot.classList.remove("disconnected");
    statusDot.classList.add("connected");
    statusText.textContent = "Connected";

    btnSend.disabled = false;
    messageInput.disabled = false;
  } catch {
    statusDot.classList.remove("connected");
    statusDot.classList.add("disconnected");
    statusText.textContent = "Disconnected";

    btnSend.disabled = true;
    messageInput.disabled = true;
  }
}

/**
 * Reads input text, renders user bubble, and dispatches to regular or streaming send.
 * @returns {Promise<void>}
 */
async function sendMessage() {
  const query = messageInput.value.trim();
  if (!query || isWaitingForResponse) {
    return;
  }

  if (!sessionId) {
    await initSession();
    if (!sessionId) {
      return;
    }
  }

  messageInput.value = "";
  autoResizeTextarea();

  welcomeCard.classList.add("hidden");
  appendUserMessage(query);
  showTypingIndicator();
  setWaitingState(true);

  if (USE_STREAMING) {
    await sendStreaming(query);
  } else {
    await sendRegular(query);
  }
}

/**
 * Sends one non-streaming message request to `/chat` and renders full response.
 * @param {string} query - User query text.
 * @returns {Promise<void>}
 */
async function sendRegular(query) {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Chat failed with ${response.status}`);
    }

    const data = await response.json();
    hideTypingIndicator();
    appendAssistantMessage(data);

    if (data.session_id) {
      sessionId = data.session_id;
    }
  } catch {
    hideTypingIndicator();
    showErrorToast("Jawab lene mein dikkat aa rahi hai. Dobara try karein.");
  } finally {
    setWaitingState(false);
  }
}

/**
 * Sends one streaming message request to `/chat/stream`, reads SSE chunks, and types them into the bubble.
 * @param {string} query - User query text.
 * @returns {Promise<void>}
 */
async function sendStreaming(query) {
  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        stream: true,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Stream failed with ${response.status}`);
    }

    hideTypingIndicator();

    const { wrapper, bubble } = createAssistantShell();
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let sseBuffer = "";
    let fullText = "";
    let doneReceived = false;

    while (!doneReceived) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      sseBuffer += decoder.decode(value, { stream: true });

      let boundaryIndex = sseBuffer.indexOf("\n\n");
      while (boundaryIndex !== -1) {
        const rawEvent = sseBuffer.slice(0, boundaryIndex);
        sseBuffer = sseBuffer.slice(boundaryIndex + 2);

        const lines = rawEvent.split("\n");
        for (const line of lines) {
          if (!line.startsWith("data:")) {
            continue;
          }

          const payload = line.slice(5).trimStart();
          if (payload === "[DONE]") {
            doneReceived = true;
            break;
          }

          fullText += payload;
          bubble.innerHTML = formatText(fullText);
          scrollToBottom();
        }

        if (doneReceived) {
          break;
        }

        boundaryIndex = sseBuffer.indexOf("\n\n");
      }
    }

    if (!fullText.trim()) {
      fullText = "Maafi chahta hoon, abhi jawab dene mein dikkat aa rahi hai.";
      bubble.innerHTML = formatText(fullText);
    }

    const looksLikeCrisis = /akele nahi hain|helpline|1860-2662-345|9152987821|108/i.test(fullText);

    addAssistantMeta(wrapper, {
      sources: [],
      language: inferLanguageFromText(fullText),
      category: looksLikeCrisis ? "crisis" : "safe",
    });

    if (looksLikeCrisis) {
      showCrisisBanner();
    }
  } catch {
    hideTypingIndicator();
    showErrorToast("Streaming mein dikkat aa rahi hai.");
  } finally {
    setWaitingState(false);
  }
}

/**
 * Appends a user message bubble on the right side.
 * @param {string} text - User text.
 */
function appendUserMessage(text) {
  const row = document.createElement("div");
  row.className = "message-row user message-enter";

  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.textContent = text;

  row.appendChild(bubble);
  chatArea.insertBefore(row, typingIndicator);
  scrollToBottom();
}

/**
 * Appends a full assistant message with metadata below the bubble.
 * @param {{answer: string, sources?: Array<{file: string, page: number|string}>, language?: string, category?: string}} response - Backend response payload.
 */
function appendAssistantMessage(response) {
  const { wrapper, bubble } = createAssistantShell();
  bubble.innerHTML = formatText(response.answer || "");

  addAssistantMeta(wrapper, {
    sources: response.sources || [],
    language: response.language || "Hindi",
    category: response.category || "safe",
  });

  if (response.category === "crisis") {
    showCrisisBanner();
  }

  scrollToBottom();
}

/**
 * Creates an assistant row skeleton used by both regular and streaming flows.
 * @returns {{wrapper: HTMLDivElement, bubble: HTMLDivElement}}
 */
function createAssistantShell() {
  const row = document.createElement("div");
  row.className = "message-row assistant message-enter";

  const wrapper = document.createElement("div");
  wrapper.className = "assistant-wrap";

  const bubble = document.createElement("div");
  bubble.className = "assistant-bubble";

  wrapper.appendChild(bubble);
  row.appendChild(wrapper);
  chatArea.insertBefore(row, typingIndicator);

  return { wrapper, bubble };
}

/**
 * Renders sources line, language badge, and fixed disclaimer under an assistant message.
 * @param {HTMLDivElement} wrapper - Assistant container element.
 * @param {{sources: Array<{file: string, page: number|string}>, language: string, category: string}} meta - Message metadata.
 */
function addAssistantMeta(wrapper, meta) {
  const metaBox = document.createElement("div");
  metaBox.className = "assistant-meta";

  const sourceLine = document.createElement("div");
  sourceLine.className = "meta-sources";

  if (meta.sources && meta.sources.length > 0) {
    const sourceText = meta.sources
      .map((source) => `${source.file} (p.${source.page})`)
      .join(", ");
    sourceLine.textContent = `📚 ${sourceText}`;
  } else {
    sourceLine.textContent = "📚 Source details stream mode mein available nahi hain";
  }

  const languageBadge = document.createElement("span");
  languageBadge.className = "lang-badge";
  languageBadge.textContent = normalizeLanguageLabel(meta.language);

  const disclaimer = document.createElement("div");
  disclaimer.className = "meta-disclaimer";
  disclaimer.textContent = "⚠️ Sirf jaankari hai, doctor ki salah zaroor lein.";

  metaBox.appendChild(sourceLine);
  metaBox.appendChild(languageBadge);
  metaBox.appendChild(disclaimer);
  wrapper.appendChild(metaBox);
}

/**
 * Shows crisis banner and keeps it visible until the user closes it manually.
 */
function showCrisisBanner() {
  crisisBanner.classList.remove("hidden");
}

/**
 * Hides crisis banner when user clicks close button.
 */
function hideCrisisBanner() {
  crisisBanner.classList.add("hidden");
}

/**
 * Makes typing indicator visible and scrolls to latest area.
 */
function showTypingIndicator() {
  typingIndicator.classList.remove("hidden");
  scrollToBottom();
}

/**
 * Hides typing indicator.
 */
function hideTypingIndicator() {
  typingIndicator.classList.add("hidden");
}

/**
 * Shows an API error toast and auto-hides it after 4 seconds.
 * @param {string} message - Toast message.
 */
function showErrorToast(message) {
  errorToastText.textContent = message;
  errorToast.classList.remove("hidden");

  requestAnimationFrame(() => {
    errorToast.classList.add("visible");
  });

  if (toastTimer) {
    clearTimeout(toastTimer);
  }

  toastTimer = setTimeout(() => {
    errorToast.classList.remove("visible");
    setTimeout(() => {
      errorToast.classList.add("hidden");
    }, 250);
  }, 4000);
}

/**
 * Clears current messages, resets session, and restores welcome state.
 * @returns {Promise<void>}
 */
async function newChat() {
  setWaitingState(false);
  hideTypingIndicator();
  hideCrisisBanner();

  if (sessionId) {
    try {
      await fetch(`${API_BASE}/session/${sessionId}`, { method: "DELETE" });
    } catch {
      // Best effort cleanup; continue creating a new session even if delete fails.
    }
  }

  chatArea.querySelectorAll(".message-row").forEach((row) => row.remove());
  welcomeCard.classList.remove("hidden");

  messageInput.value = "";
  autoResizeTextarea();

  await initSession();
  await checkHealth();
  messageInput.focus();
}

/**
 * Auto-resizes textarea to content height up to 120px (about 4 lines).
 */
function autoResizeTextarea() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 120)}px`;
}

/**
 * Escapes unsafe HTML and converts line breaks to `<br>`.
 * @param {string} text - Raw assistant text.
 * @returns {string}
 */
function formatText(text) {
  const escaped = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return escaped.replace(/\n/g, "<br>");
}

/**
 * Normalizes language values into UI labels.
 * @param {string} language - Raw language string.
 * @returns {string}
 */
function normalizeLanguageLabel(language) {
  const value = (language || "hindi").toLowerCase();

  if (value.includes("hinglish")) {
    return "Hinglish";
  }
  if (value.includes("english")) {
    return "English";
  }
  return "Hindi";
}

/**
 * Infers response language from text in streaming mode when metadata is unavailable.
 * @param {string} text - Assistant response text.
 * @returns {string}
 */
function inferLanguageFromText(text) {
  const hasHindi = /[\u0900-\u097F]/.test(text);
  const hasEnglish = /[A-Za-z]/.test(text);

  if (hasHindi && hasEnglish) {
    return "hinglish";
  }
  if (hasEnglish) {
    return "english";
  }
  return "hindi";
}

/**
 * Sets waiting state and toggles send button/input interactivity.
 * @param {boolean} waiting - Whether an API request is in progress.
 */
function setWaitingState(waiting) {
  isWaitingForResponse = waiting;
  btnSend.disabled = waiting || statusText.textContent === "Disconnected";
  messageInput.disabled = waiting || statusText.textContent === "Disconnected";
}

/**
 * Scrolls chat area to the bottom smoothly after each update.
 */
function scrollToBottom() {
  chatArea.scrollTo({
    top: chatArea.scrollHeight,
    behavior: "smooth",
  });
}
