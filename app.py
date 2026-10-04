"""
VoiceAI Pro — Whisper + Offline Assistant + gTTS
Run:  streamlit run voiceai_pro.py
Needs: pip install -r requirements.txt
       FFmpeg is bundled automatically by imageio-ffmpeg
"""

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from typing import Dict, Optional, Union

import streamlit as st
import whisper
from gtts import gTTS
from gtts.lang import tts_langs

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

# =========================================
# Page Config
# =========================================
st.set_page_config(
    page_title="VoiceAI Pro",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# Styling
# =========================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary:#080e1a; --bg-card:#0d1526; --bg-glass:rgba(13,21,38,.85);
    --accent:#00d4ff; --accent2:#7b5ea7; --success:#00e5a0; --warning:#ffb547;
    --text-primary:#e8f0ff; --text-muted:#6a7fa8; --border:rgba(0,212,255,.15);
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text-primary); }
.stApp {
    background-color:var(--bg-primary);
    background-image:
        linear-gradient(rgba(0,212,255,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,.03) 1px, transparent 1px);
    background-size:40px 40px;
}
#MainMenu, footer { visibility:hidden; }
header[data-testid="stHeader"] { background:transparent; }

[data-testid="stSidebar"] { background:var(--bg-card); border-right:1px solid var(--border); }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family:'Space Mono',monospace; color:var(--accent); font-size:.85rem;
    letter-spacing:.2em; text-transform:uppercase; margin-bottom:1rem;
}
.hero-title {
    font-family:'Space Mono',monospace; font-size:2.6rem; font-weight:700;
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    letter-spacing:-.02em; margin-bottom:0;
}
.hero-sub { color:var(--text-muted); font-size:.95rem; font-weight:300; letter-spacing:.05em; margin-top:4px; }

.waveform { display:flex; align-items:center; gap:4px; height:36px; margin:16px 0; justify-content:flex-end; }
.waveform span { display:inline-block; width:3px; border-radius:3px; background:var(--accent);
    animation:wave 1.2s ease-in-out infinite; opacity:.6; }
.waveform span:nth-child(1){height:8px} .waveform span:nth-child(2){height:20px;animation-delay:.1s}
.waveform span:nth-child(3){height:30px;animation-delay:.2s} .waveform span:nth-child(4){height:24px;animation-delay:.3s}
.waveform span:nth-child(5){height:36px;animation-delay:.4s} .waveform span:nth-child(6){height:28px;animation-delay:.3s}
.waveform span:nth-child(7){height:18px;animation-delay:.2s} .waveform span:nth-child(8){height:10px;animation-delay:.1s}
.waveform span:nth-child(9){height:22px} .waveform span:nth-child(10){height:14px;animation-delay:.15s}
@keyframes wave { 0%,100%{transform:scaleY(1);opacity:.6} 50%{transform:scaleY(1.8);opacity:1} }

.section-label { font-family:'Space Mono',monospace; font-size:.72rem; letter-spacing:.18em;
    text-transform:uppercase; color:var(--text-muted); margin-bottom:8px; }

.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:.75rem;
    font-weight:600; letter-spacing:.05em; margin-right:6px; }
.badge-success { background:rgba(0,229,160,.12); color:var(--success); border:1px solid rgba(0,229,160,.3); }
.badge-info    { background:rgba(0,212,255,.1);  color:var(--accent);  border:1px solid rgba(0,212,255,.25); }
.badge-warn    { background:rgba(255,181,71,.1); color:var(--warning); border:1px solid rgba(255,181,71,.25); }

.stButton > button, .stDownloadButton > button {
    background:linear-gradient(135deg,#00d4ff22,#7b5ea722); color:var(--accent) !important;
    border:1px solid var(--accent) !important; border-radius:10px !important;
    font-family:'Space Mono',monospace !important; font-size:.8rem !important;
    letter-spacing:.05em !important; min-height:42px !important; transition:all .2s ease !important; width:100%;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background:linear-gradient(135deg,#00d4ff44,#7b5ea744) !important;
    transform:translateY(-1px); box-shadow:0 4px 20px rgba(0,212,255,.15) !important;
}
[data-testid="stChatMessage"] {
    background:var(--bg-card); border:1px solid var(--border); border-radius:14px; padding:10px 14px;
}
[data-testid="stFileUploader"] { background:var(--bg-card) !important; border:2px dashed var(--border) !important;
    border-radius:14px !important; padding:12px !important; }
hr { border-color:var(--border) !important; opacity:.4; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg-primary); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================
# Constants
# =========================================
WHISPER_MODELS = ["tiny", "base", "small", "medium", "turbo", "large"]

# label -> whisper code (None = auto-detect)
STT_LANGS = {
    "Auto-detect": None, "English": "en", "Urdu": "ur", "Arabic": "ar", "French": "fr",
    "Spanish": "es", "German": "de", "Hindi": "hi", "Turkish": "tr", "Chinese": "zh", "Japanese": "ja",
}
# whisper code -> gTTS code where they differ
WHISPER_TO_GTTS = {"zh": "zh-CN"}
LANG_NAMES = {v: k for k, v in STT_LANGS.items() if v}

SUPPORTED_TTS = tts_langs()


def to_tts_code(code: Optional[str]) -> str:
    code = WHISPER_TO_GTTS.get(code or "en", code or "en")
    return code if code in SUPPORTED_TTS else "en"


# =========================================
# Session State
# =========================================
ss = st.session_state
ss.setdefault("chat_history", [])        # [{"user","assistant","latency","tokens"}]
ss.setdefault("message", "")
ss.setdefault("last_audio_id", None)
ss.setdefault("transcript_info", None)   # last transcription result dict
if ss.pop("_clear_message", False):
    ss["message"] = ""

# =========================================
# Sidebar
# =========================================
with st.sidebar:
    st.markdown("## ⚙ Speech Recognition")
    model_size = st.selectbox("Whisper model", WHISPER_MODELS, index=2,
                              help="Larger = more accurate but slower. 'small' is a good default; 'turbo' is fast and accurate on GPU.")
    stt_label = st.selectbox("Spoken language", list(STT_LANGS.keys()), index=0,
                             help="Setting this explicitly is more accurate than auto-detect for short clips.")
    stt_lang = STT_LANGS[stt_label]
    task = st.radio("Task", ["Transcribe", "Translate to English"], horizontal=True)
    vocab_hint = st.text_input("Vocabulary hints", placeholder="Names, jargon, acronyms…",
                               help="Passed to Whisper as a prompt to improve spelling of rare words.")

    st.markdown("---")
    st.markdown("## 🔊 Voice Output")
    auto_tts = st.toggle("Speak replies", value=True)
    tts_choice = st.selectbox("TTS language", ["Auto (follow input)"] + [k for k in STT_LANGS if k != "Auto-detect"])
    tts_speed = st.select_slider("Voice speed", ["Slow", "Normal"], value="Normal")
    slow_tts = tts_speed == "Slow"

    st.markdown("---")
    st.markdown("## 🤖 Offline Assistant")
    temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)
    max_tokens = st.slider("Max reply tokens", 100, 4000, 800, 100)
    memory_turns = st.slider("Memory (turns)", 0, 20, 8)
    reply_in_user_lang = st.checkbox("Reply in the user's language", value=True)
    system_prompt = st.text_area(
        "System prompt",
        value="You are VoiceAI Pro, a helpful and concise voice assistant. Keep responses under 3 sentences. "
              "Avoid markdown, lists and emojis because replies are spoken aloud.",
        height=110,
    )
    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        ss.chat_history = []
        st.rerun()

    st.markdown(
        f"<div style='color:#6a7fa8;font-size:.75rem;'>Device: {'GPU (CUDA)' if HAS_CUDA else 'CPU'}</div>",
        unsafe_allow_html=True,
    )

def ensure_ffmpeg() -> bool:
    """Make FFmpeg available to Whisper, including when bundled by imageio-ffmpeg."""
    if shutil.which("ffmpeg"):
        return True

    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError):
        return False

    import whisper.audio as whisper_audio

    original_run = whisper_audio.run

    def run_ffmpeg(command, *args, **kwargs):
        if command and command[0] == "ffmpeg":
            command = [ffmpeg_path, *command[1:]]
        return original_run(command, *args, **kwargs)

    whisper_audio.run = run_ffmpeg
    return os.path.isfile(ffmpeg_path)


if not ensure_ffmpeg():
    st.error(
        "FFmpeg was not found. Install the app dependencies with "
        "`pip install -r requirements.txt`, then restart Streamlit."
    )
    st.stop()


# =========================================
# Core helpers
# =========================================
@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper(size: str):
    load_model = getattr(whisper, "load_model", None)
    if load_model is None:
        raise RuntimeError(
            "The installed whisper package is not OpenAI Whisper. "
            "Install it with: pip install -U openai-whisper"
        )
    return load_model(size, device="cuda" if HAS_CUDA else "cpu")


@st.cache_data(show_spinner=False, max_entries=32)
def transcribe_audio(audio_bytes: bytes, suffix: str, size: str, language, task_name: str, prompt: str) -> dict:
    """Cached by content hash + settings, so Streamlit reruns never re-transcribe."""
    model = load_whisper(size)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        t0 = time.time()
        result = model.transcribe(
            path,
            language=language,
            task="translate" if task_name.startswith("Translate") else "transcribe",
            initial_prompt=prompt or None,
            fp16=HAS_CUDA,
            condition_on_previous_text=False,  # reduces repetition/hallucination loops
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
        )
        elapsed = time.time() - t0
    finally:
        os.unlink(path)

    segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]
    return {
        "text": result["text"].strip(),
        "language": result.get("language", language or "en"),
        "segments": segments,
        "duration": segments[-1]["end"] if segments else 0.0,
        "elapsed": elapsed,
    }


def fmt_ts(sec: float, srt: bool = False) -> str:
    ms = int((sec - int(sec)) * 1000)
    s = int(sec)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{s:02d}{',' if srt else '.'}{ms:03d}"


def to_srt(segments: list) -> str:
    return "\n".join(
        f"{i}\n{fmt_ts(s['start'], True)} --> {fmt_ts(s['end'], True)}\n{s['text']}\n"
        for i, s in enumerate(segments, 1)
    )


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_`#>~]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_data(show_spinner=False, max_entries=64)
def synthesize(text: str, lang: str, slow: bool) -> bytes:
    buf = io.BytesIO()
    gTTS(text=strip_markdown(text), lang=lang, slow=slow).write_to_fp(buf)
    return buf.getvalue()


def resolve_tts_lang() -> str:
    if tts_choice != "Auto (follow input)":
        return to_tts_code(STT_LANGS[tts_choice])
    info = ss.get("transcript_info")
    return to_tts_code(info["language"] if info else (stt_lang or "en"))


def build_messages(user_text: str) -> list:
    msgs = []
    for turn in ss.chat_history[-memory_turns:] if memory_turns else []:
        msgs.append({"role": "user", "content": turn["user"]})
        msgs.append({"role": "assistant", "content": turn["assistant"]})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def stream_offline(
    user_text: str, placeholder
) -> tuple[str, Dict[str, Union[float, int]]]:
    t0 = time.time()
    normalized = user_text.lower().strip()
    if normalized in {"hi", "hello", "hey", "salam", "assalam o alaikum"}:
        reply = "Hello! How can I help you?"
    elif "time" in normalized:
        reply = f"The current local time is {time.strftime('%I:%M %p')}."
    elif "help" in normalized:
        reply = "I can transcribe audio, translate speech, and convert text to speech. Type a message or upload an audio file to begin."
    elif normalized.endswith("?"):
        reply = "I am running in offline mode, so I cannot search the internet or use an online AI model. I can still help with transcription and text-to-speech."
    else:
        reply = f"I received your message: {user_text}"
    placeholder.markdown(reply)
    usage: Dict[str, Union[float, int]] = {
        "latency": time.time() - t0,
        "in_tokens": 0,
        "out_tokens": 0,
    }
    return reply, usage


def chat_to_markdown() -> str:
    return "\n\n".join(f"**You:** {t['user']}\n\n**Assistant:** {t['assistant']}" for t in ss.chat_history)


# =========================================
# Header
# =========================================
col_title, col_wave = st.columns([3, 1])
with col_title:
    st.markdown('<div class="hero-title">VoiceAI Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Speech · Offline Assistant · Voice — Powered by Whisper &amp; gTTS</div>',
                unsafe_allow_html=True)
with col_wave:
    st.markdown('<div class="waveform">' + "<span></span>" * 10 + "</div>", unsafe_allow_html=True)
st.markdown("---")

left_col, right_col = st.columns(2, gap="large")

# =========================================
# LEFT — Input
# =========================================
send_requested = False
with left_col:
    st.markdown('<div class="section-label">🎧 Audio Input</div>', unsafe_allow_html=True)
    tab_upload, tab_record = st.tabs(["📂 Upload", "🎤 Record"])

    audio_file = None
    with tab_upload:
        up = st.file_uploader("Audio file", type=["mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"],
                              label_visibility="collapsed")
        if up:
            audio_file = up
    with tab_record:
        if hasattr(st, "audio_input"):
            rec = st.audio_input("Record a message", label_visibility="collapsed")
            if rec:
                audio_file = rec
        else:
            st.info("Upgrade Streamlit (`pip install -U streamlit`) to enable microphone recording.")

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        suffix = os.path.splitext(getattr(audio_file, "name", "") or "")[1] or ".wav"
        st.audio(audio_bytes)

        audio_id = hashlib.sha1(
            audio_bytes + f"{model_size}|{stt_lang}|{task}|{vocab_hint}".encode()
        ).hexdigest()

        with st.spinner(f"🔍 Transcribing with Whisper ({model_size})…"):
            info = transcribe_audio(audio_bytes, suffix, model_size, stt_lang, task, vocab_hint)
        ss.transcript_info = info

        # Only overwrite the editable message when the audio/settings actually changed
        if ss.last_audio_id != audio_id:
            ss.last_audio_id = audio_id
            ss["message"] = info["text"]

        lang_name = LANG_NAMES.get(info["language"], info["language"].upper())
        st.markdown(
            f'<span class="badge badge-success">✓ Transcribed</span>'
            f'<span class="badge badge-info">🌐 {lang_name}</span>'
            f'<span class="badge badge-info">⏱ {info["duration"]:.1f}s audio · {info["elapsed"]:.1f}s to process</span>',
            unsafe_allow_html=True,
        )
        if not info["text"]:
            st.warning("No speech detected. Try a clearer recording or a larger Whisper model.")

        with st.expander("Timestamped segments & export"):
            for s in info["segments"]:
                st.markdown(f"`{fmt_ts(s['start'])}` {s['text']}")
            d1, d2, d3 = st.columns(3)
            d1.download_button("TXT", info["text"], "transcript.txt", use_container_width=True)
            d2.download_button("SRT", to_srt(info["segments"]), "transcript.srt", use_container_width=True)
            d3.download_button("JSON", json.dumps(info, ensure_ascii=False, indent=2),
                               "transcript.json", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-label">✏️ Message (edit transcript or type your own)</div>',
                unsafe_allow_html=True)
    st.text_area("Message", key="message", placeholder="Ask anything…", height=130,
                 label_visibility="collapsed")
    if st.button("🚀 Send", use_container_width=True):
        send_requested = True

# =========================================
# RIGHT — Conversation
# =========================================
with right_col:
    st.markdown('<div class="section-label">💬 Conversation</div>', unsafe_allow_html=True)

    tts_lang_code = resolve_tts_lang()
    n = len(ss.chat_history)

    if not ss.chat_history and not send_requested:
        st.markdown('<div style="color:#6a7fa8;font-size:.9rem;padding:20px 0;">Conversation will appear here…</div>',
                    unsafe_allow_html=True)

    for i, turn in enumerate(ss.chat_history):
        with st.chat_message("user", avatar="👤"):
            st.markdown(turn["user"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(turn["assistant"])
            st.caption(f"{turn['latency']:.1f}s · {turn['in_tokens']} in / {turn['out_tokens']} out tokens")
            if auto_tts and i == n - 1:
                try:
                    st.audio(synthesize(turn["assistant"], turn.get("tts_lang", tts_lang_code), slow_tts),
                             format="audio/mp3")
                except Exception as e:
                    st.warning(f"Voice generation failed: {e}")
            elif auto_tts and st.button("🔊 Play", key=f"play_{i}"):
                st.audio(synthesize(turn["assistant"], turn.get("tts_lang", tts_lang_code), slow_tts),
                         format="audio/mp3")

    if send_requested:
        user_msg = ss["message"].strip()
        if not user_msg:
            st.warning("Please upload/record audio or type a message first.")
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_msg)
            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                try:
                    reply, stats = stream_offline(user_msg, placeholder)
                except RuntimeError as e:
                    placeholder.error(str(e))
                else:
                    ss.chat_history.append({"user": user_msg, "assistant": reply, "tts_lang": tts_lang_code, **stats})
                    ss["_clear_message"] = True
                    st.rerun()

    if ss.chat_history:
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Export chat (MD)", chat_to_markdown(), "conversation.md", use_container_width=True)
        c2.download_button("⬇ Export chat (JSON)", json.dumps(ss.chat_history, ensure_ascii=False, indent=2),
                           "conversation.json", use_container_width=True)

# =========================================
# Standalone TTS
# =========================================
st.markdown("---")
st.markdown('<div class="section-label">🗣️ Standalone Text → Speech</div>', unsafe_allow_html=True)
t1, t2 = st.columns([2, 1])
with t1:
    tts_input = st.text_area("Text to speak", placeholder="Type something to hear it spoken…",
                             height=90, label_visibility="collapsed", key="tts_input")
with t2:
    st.markdown("<br>", unsafe_allow_html=True)
    convert_btn = st.button("🎙️ Generate Voice", use_container_width=True)

if convert_btn:
    if tts_input.strip():
        try:
            with st.spinner("Generating voice…"):
                audio = synthesize(tts_input.strip(), resolve_tts_lang(), slow_tts)
            st.markdown('<span class="badge badge-success">✓ Voice Ready</span>', unsafe_allow_html=True)
            st.audio(audio, format="audio/mp3")
            st.download_button("⬇ Download MP3", audio, "voice.mp3", mime="audio/mpeg")
        except Exception as e:
            st.error(f"Voice generation failed (gTTS needs internet access): {e}")
    else:
        st.warning("Please enter some text first.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#6a7fa8;font-size:.8rem;padding-bottom:10px;"
    "font-family:Space Mono,monospace;'>VOICEAI PRO &nbsp;·&nbsp; Whisper · Offline Assistant · gTTS</div>",
    unsafe_allow_html=True,
)
