import streamlit as st
import whisper
from gtts import gTTS
from gtts.lang import tts_langs
import tempfile
import os
import time
import anthropic
import pyperclip

# =========================================
# Page Config
# =========================================
st.set_page_config(
    page_title="VoiceAI Pro",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# Premium Dark UI Styling
# =========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:   #080e1a;
    --bg-card:      #0d1526;
    --bg-glass:     rgba(13, 21, 38, 0.85);
    --accent:       #00d4ff;
    --accent2:      #7b5ea7;
    --success:      #00e5a0;
    --warning:      #ffb547;
    --text-primary: #e8f0ff;
    --text-muted:   #6a7fa8;
    --border:       rgba(0, 212, 255, 0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

/* Animated background grid */
.stApp {
    background-color: var(--bg-primary);
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Main title */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Waveform animation */
.waveform {
    display: flex;
    align-items: center;
    gap: 4px;
    height: 36px;
    margin: 16px 0;
}
.waveform span {
    display: inline-block;
    width: 3px;
    border-radius: 3px;
    background: var(--accent);
    animation: wave 1.2s ease-in-out infinite;
    opacity: 0.6;
}
.waveform span:nth-child(1)  { height: 8px;  animation-delay: 0s; }
.waveform span:nth-child(2)  { height: 20px; animation-delay: 0.1s; }
.waveform span:nth-child(3)  { height: 30px; animation-delay: 0.2s; }
.waveform span:nth-child(4)  { height: 24px; animation-delay: 0.3s; }
.waveform span:nth-child(5)  { height: 36px; animation-delay: 0.4s; }
.waveform span:nth-child(6)  { height: 28px; animation-delay: 0.3s; }
.waveform span:nth-child(7)  { height: 18px; animation-delay: 0.2s; }
.waveform span:nth-child(8)  { height: 10px; animation-delay: 0.1s; }
.waveform span:nth-child(9)  { height: 22px; animation-delay: 0s; }
.waveform span:nth-child(10) { height: 14px; animation-delay: 0.15s; }
@keyframes wave {
    0%, 100% { transform: scaleY(1); opacity: 0.6; }
    50%       { transform: scaleY(1.8); opacity: 1; }
}

/* Cards */
.glass-card {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}

/* Transcript / response boxes */
.result-box {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(123,94,167,0.05));
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 16px 18px;
    color: var(--text-primary);
    font-size: 1rem; run 
    line-height: 1.6;
    margin-top: 8px;
}
.result-box.assistant {
    border-left-color: var(--accent2);
}

/* Chat bubbles */
.chat-bubble {
    display: flex;
    gap: 12px;
    margin-bottom: 14px;
    align-items: flex-start;
}
.chat-bubble .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.chat-bubble.user .avatar  { background: rgba(0,212,255,0.15); color: var(--accent); }
.chat-bubble.ai .avatar    { background: rgba(123,94,167,0.2); color: var(--accent2); }
.chat-bubble .bubble-text {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 10px 14px;
    font-size: 0.9rem;
    line-height: 1.5;
    flex: 1;
    border: 1px solid var(--border);
}
.chat-bubble.user .bubble-text  { border-color: rgba(0,212,255,0.2); }
.chat-bubble.ai .bubble-text    { border-color: rgba(123,94,167,0.2); }

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 8px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-success { background: rgba(0,229,160,0.12); color: var(--success); border: 1px solid rgba(0,229,160,0.3); }
.badge-info    { background: rgba(0,212,255,0.1);  color: var(--accent);  border: 1px solid rgba(0,212,255,0.25); }
.badge-warn    { background: rgba(255,181,71,0.1);  color: var(--warning); border: 1px solid rgba(255,181,71,0.25); }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff22, #7b5ea722);
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    height: 42px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff44, #7b5ea744) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,212,255,0.15) !important;
}

/* Selectbox & textarea */
.stSelectbox > div > div, .stTextArea > div > textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

/* Divider */
hr { border-color: var(--border) !important; opacity: 0.4; }

/* Slider */
.stSlider > div { color: var(--text-muted) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# =========================================
# Session State Init
# =========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "transcript" not in st.session_state:
    st.session_state.transcript = ""


# =========================================
# Sidebar — Settings
# =========================================
with st.sidebar:
    st.markdown("## ⚙ Settings")

    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium"],
        index=0,
        help="Larger = more accurate but slower"
    )

    LANG_OPTIONS = {
        "English": "en", "Urdu": "ur", "Arabic": "ar",
        "French": "fr", "Spanish": "es", "German": "de",
        "Hindi": "hi", "Turkish": "tr", "Chinese": "zh-CN", "Japanese": "ja"
    }
    tts_lang_label = st.selectbox("TTS Language", list(LANG_OPTIONS.keys()), index=0)
    tts_lang = LANG_OPTIONS[tts_lang_label]

    tts_speed = st.select_slider(
        "Voice Speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal"
    )
    slow_tts = tts_speed == "Slow"

    st.markdown("---")
    st.markdown("## 🤖 AI Model")
    system_prompt = st.text_area(
        "System Prompt",
        value="You are VoiceAI Pro, a helpful and concise voice assistant. Keep responses under 3 sentences.",
        height=100
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='color:#6a7fa8;font-size:0.75rem;'>"
        "VoiceAI Pro · Whisper + Claude + gTTS"
        "</div>",
        unsafe_allow_html=True
    )


# =========================================
# Load Whisper Model (cached)
# =========================================
@st.cache_resource
def load_whisper(size):
    return whisper.load_model(size)

model = load_whisper(model_size)


# =========================================
# Claude API Response
# =========================================
def get_claude_response(user_text: str, history: list, sys_prompt: str) -> str:
    client = anthropic.Anthropic()
    messages = []
    for turn in history[-6:]:  # last 6 turns for context
        messages.append({"role": "user",      "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_text})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=sys_prompt,
        messages=messages
    )
    return response.content[0].text


# =========================================
# TTS Helper
# =========================================
def text_to_speech(text: str, lang: str, slow: bool) -> bytes:
    tts = gTTS(text=text, lang=lang, slow=slow)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        with open(f.name, "rb") as af:
            audio_bytes = af.read()
    os.unlink(f.name)
    return audio_bytes


# =========================================
# Header
# =========================================
col_title, col_wave = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">VoiceAI Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Speech · Intelligence · Voice — Powered by Whisper & Claude</div>', unsafe_allow_html=True)
with col_wave:
    st.markdown(
        '<div class="waveform" style="justify-content:flex-end">'
        + ''.join(['<span></span>'] * 10) +
        '</div>',
        unsafe_allow_html=True
    )

st.markdown("---")


# =========================================
# Layout: Two Columns
# =========================================
left_col, right_col = st.columns([1, 1], gap="large")


# ─── LEFT: Input ──────────────────────────
with left_col:

    # --- Audio Upload ---
    st.markdown('<div class="section-label">📂 Audio Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop an MP3 / WAV / M4A file here",
        type=["mp3", "wav", "m4a"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🔍 Transcribing audio…"):
            result = model.transcribe(tmp_path)
            st.session_state.transcript = result["text"]
        os.unlink(tmp_path)

        st.markdown(
            '<span class="badge badge-success">✓ Transcribed</span>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="section-label" style="margin-top:12px">Transcript</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-box">{st.session_state.transcript}</div>',
            unsafe_allow_html=True
        )

        if st.button("📋 Copy Transcript"):
            try:
                pyperclip.copy(st.session_state.transcript)
                st.success("Copied!")
            except Exception:
                st.code(st.session_state.transcript)

    st.markdown("---")

    # --- Text Input ---
    st.markdown('<div class="section-label">✏️ Or Type Your Message</div>', unsafe_allow_html=True)
    manual_input = st.text_area(
        "Type here",
        placeholder="Ask anything…",
        height=100,
        label_visibility="collapsed"
    )

    use_transcript = st.checkbox(
        "Use transcript as input",
        value=bool(st.session_state.transcript)
    )

    send_btn = st.button("🚀 Send to Claude", use_container_width=True)


# ─── RIGHT: Output ────────────────────────
with right_col:

    st.markdown('<div class="section-label">💬 Conversation</div>', unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        for turn in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-bubble user">'
                f'  <div class="avatar">👤</div>'
                f'  <div class="bubble-text">{turn["user"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="chat-bubble ai">'
                f'  <div class="avatar">🤖</div>'
                f'  <div class="bubble-text">{turn["assistant"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<div style="color:#6a7fa8;font-size:0.9rem;padding:20px 0;">'
            'Conversation will appear here…'
            '</div>',
            unsafe_allow_html=True
        )

    # Process send
    if send_btn:
        user_msg = (st.session_state.transcript if use_transcript else manual_input).strip()
        if user_msg:
            with st.spinner("🤖 Claude is thinking…"):
                reply = get_claude_response(user_msg, st.session_state.chat_history, system_prompt)

            st.session_state.chat_history.append({"user": user_msg, "assistant": reply})

            # TTS for response
            st.markdown("---")
            st.markdown('<div class="section-label">🔊 Assistant Voice</div>', unsafe_allow_html=True)
            audio_bytes = text_to_speech(reply, tts_lang, slow_tts)
            st.audio(audio_bytes, format="audio/mp3")

            if st.button("📋 Copy Response"):
                try:
                    pyperclip.copy(reply)
                    st.success("Copied!")
                except Exception:
                    st.code(reply)

            st.rerun()
        else:
            st.warning("Please upload audio or type a message first.")


# =========================================
# Text-to-Speech Section (standalone)
# =========================================
st.markdown("---")
st.markdown('<div class="section-label">🗣️ Standalone Text → Speech</div>', unsafe_allow_html=True)

tts_col1, tts_col2 = st.columns([2, 1])
with tts_col1:
    tts_input = st.text_area(
        "Enter any text to convert to voice",
        placeholder="Type something to hear it spoken…",
        height=90,
        label_visibility="collapsed"
    )
with tts_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    convert_btn = st.button("🎙️ Generate Voice", use_container_width=True)

if convert_btn:
    if tts_input.strip():
        with st.spinner("Generating voice…"):
            tts_audio = text_to_speech(tts_input.strip(), tts_lang, slow_tts)
        st.markdown('<span class="badge badge-success">✓ Voice Ready</span>', unsafe_allow_html=True)
        st.audio(tts_audio, format="audio/mp3")
    else:
        st.warning("Please enter some text first.")


# =========================================
# Footer
# =========================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6a7fa8;font-size:0.8rem;padding-bottom:10px;font-family:Space Mono,monospace;'>"
    "VOICEAI PRO &nbsp;·&nbsp; Whisper · Claude · gTTS"
    "</div>",
    unsafe_allow_html=True
)
