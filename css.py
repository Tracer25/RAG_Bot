
UI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-main: #f4f7f3;
  --bg-panel: #ffffff;
  --ink-main: #0d1b1e;
  --ink-subtle: #506066;
  --accent: #0f766e;
  --accent-soft: #d9f6ee;
  --outline: #d5ddd8;
}

.gradio-container {
  background:
    radial-gradient(circle at 5% -5%, #d8f5eb 0%, transparent 34%),
    radial-gradient(circle at 100% 0%, #d6ecff 0%, transparent 30%),
    var(--bg-main);
  font-family: "Space Grotesk", sans-serif;
}

#title-wrap {
  background: linear-gradient(135deg, #0f766e, #1d4ed8);
  color: #f8fffd;
  border-radius: 18px;
  padding: 1rem 1.2rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 14px 34px rgba(15, 118, 110, 0.26);
  margin-bottom: 8px;
}

.panel {
  background: var(--bg-panel);
  border: 1px solid var(--outline);
  border-radius: 16px;
}

#chat-box {
  border: 1px solid var(--outline);
  border-radius: 14px;
}

#source-box textarea,
#meta-box textarea {
  font-family: "IBM Plex Mono", monospace;
}
"""