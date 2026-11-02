import os
import time
import uuid
import httpx
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Zangoh AI Support Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Minimalist Dark CSS (Linear / Vercel Aesthetic)
st.markdown(
    """
    <style>
    /* Global font & background adjustments */
    .stApp {
        background-color: #090a0f;
        color: #f4f4f5;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* PREVENT ALL STREAMLIT SCREEN DIMMING / DARKENING WHEN PROCESSING */
    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"],
    .element-container,
    [data-testid="stVerticalBlock"],
    .stChatMessage,
    .stMarkdown,
    .stButton,
    .stApp {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }

    /* ELIMINATE SIDEBAR UNWANTED SCROLLBAR */
    section[data-testid="stSidebar"] {
        scrollbar-width: none !important;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        display: none !important;
    }
    div[data-testid="stSidebarContent"] {
        scrollbar-width: none !important;
        padding-top: 1.25rem !important;
        padding-bottom: 1.25rem !important;
    }
    div[data-testid="stSidebarContent"]::-webkit-scrollbar {
        display: none !important;
    }

    /* Subtle Header Styling */
    .app-header {
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .app-title {
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .app-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Source Badge Styling */
    .source-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px;
    }
    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(59, 130, 246, 0.08);
        color: #93c5fd;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }

    /* Ticket Card Styling */
    .ticket-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 10px;
        color: #34d399;
        font-size: 0.88rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .ticket-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 4px;
    }
    .ticket-badge-open {
        background: rgba(16, 185, 129, 0.25);
        color: #10b981;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }

    /* Quick Toolbar Header */
    .starter-header {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #71717a;
        margin-bottom: 0.5rem;
    }
    
    /* Telemetry Compact Pill */
    .telemetry-pill-compact {
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 0.73rem;
        color: #cbd5e1;
        background: rgba(255, 255, 255, 0.03);
        padding: 4px 7px;
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Modern In-Chat Spinner Styling */
    div.stSpinner > div {
        border-top-color: #38bdf8 !important;
        border-right-color: rgba(56, 189, 248, 0.2) !important;
        border-bottom-color: rgba(56, 189, 248, 0.2) !important;
        border-left-color: rgba(56, 189, 248, 0.2) !important;
    }
    div.stSpinner {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        padding: 4px 0 !important;
    }

    /* Sleek Dark Mode HTML5 Audio Player */
    audio {
        width: 100% !important;
        height: 38px !important;
        border-radius: 6px !important;
        filter: invert(0.88) hue-rotate(180deg) !important;
        margin-top: 8px !important;
        outline: none !important;
    }

    /* Voice expander styling */
    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        margin-top: 10px !important;
        margin-bottom: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_ticket" not in st.session_state:
    st.session_state.latest_ticket = None
if "next_query" not in st.session_state:
    st.session_state.next_query = None
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "active_audio" not in st.session_state:
    st.session_state.active_audio = None
if "pending_transcript" not in st.session_state:
    st.session_state.pending_transcript = None
if "last_transcribed_audio_hash" not in st.session_state:
    st.session_state.last_transcribed_audio_hash = None
if "stt_error" not in st.session_state:
    st.session_state.stt_error = None
if "tts_errors" not in st.session_state:
    st.session_state.tts_errors = {}
if "recorder_key" not in st.session_state:
    st.session_state.recorder_key = 0

MAX_TTS_CHARS = 4000


def get_error_message(response: httpx.Response, fallback: str) -> str:
    """Read the API's error text, whatever shape the error body has."""
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("error") or body.get("detail") or fallback)
    except Exception:
        pass
    return fallback


def synthesize_message(msg_id: str, text: str) -> bool:
    """Generate speech for one response on demand and cache it. Returns True on success."""
    if msg_id in st.session_state.audio_cache:
        return True
    try:
        res = httpx.post(
            f"{API_BASE_URL}/voice/synthesize",
            json={"message_id": msg_id, "text": text[:MAX_TTS_CHARS]},
            timeout=90.0,
        )
        if res.status_code == 200 and res.content:
            st.session_state.audio_cache[msg_id] = res.content
            st.session_state.tts_errors.pop(msg_id, None)
            return True
        st.session_state.tts_errors[msg_id] = get_error_message(res, f"Voice service error (HTTP {res.status_code})")
    except httpx.ConnectError:
        st.session_state.tts_errors[msg_id] = f"Could not reach the API server at {API_BASE_URL}."
    except Exception as exc:
        st.session_state.tts_errors[msg_id] = f"Voice playback failed: {exc}"
    return False

def queue_prompt(prompt_text: str):
    st.session_state.next_query = prompt_text

def reset_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.latest_ticket = None
    st.session_state.next_query = None
    st.session_state.audio_cache = {}
    st.session_state.active_audio = None
    st.session_state.pending_transcript = None
    st.session_state.last_transcribed_audio_hash = None
    st.session_state.stt_error = None
    st.session_state.tts_errors = {}
    # Recreate the microphone widget so the old recording is discarded
    st.session_state.recorder_key += 1
    # Clear leftover widget values
    st.session_state.pop("transcript_editor_box", None)
    st.session_state["ticket_lookup_input"] = ""

# ==============================================================================
# SIDEBAR: COMPACT SYSTEM TELEMETRY & TICKET INSPECTOR (ZERO SCROLLBAR)
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Agent Control")

    # Backend Health Check
    try:
        health_resp = httpx.get(f"{API_BASE_URL}/health", timeout=2.0)
        if health_resp.status_code == 200:
            st.success("Backend: Connected (Port 8000)", icon="🟢")
        else:
            st.warning(f"Backend: HTTP {health_resp.status_code}", icon="🟡")
    except Exception:
        st.error("Backend: Offline (Port 8000)", icon="🔴")

    st.markdown("##### Telemetry")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="telemetry-pill-compact">⚡ Qwen2.5 3B (GPU)</div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-pill-compact">🎙️ Whisper (STT)</div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-pill-compact">🔄 LangGraph Core</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="telemetry-pill-compact">🔊 Edge TTS (Neural)</div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-pill-compact">📚 ChromaDB RAG</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="telemetry-pill-compact">🆔 {st.session_state.session_id[:6]}...</div>', unsafe_allow_html=True)

    st.button("🔄 Reset Conversation", key="sidebar_reset_btn", on_click=reset_chat, use_container_width=True)

    st.divider()

    # TICKET INSPECTOR WIDGET (GET /tickets/{id})
    st.markdown("##### 🎫 Ticket Inspector")
    if st.session_state.latest_ticket:
        st.caption(f"Active: **`{st.session_state.latest_ticket}`**")

    lookup_id = st.text_input(
        "Ticket ID",
        value=st.session_state.latest_ticket or "",
        placeholder="e.g. CST-2026-0001",
        key="ticket_lookup_input",
        label_visibility="collapsed",
    )
    if st.button("🔍 Fetch Ticket Record", key="ticket_lookup_btn", use_container_width=True):
        if lookup_id.strip():
            try:
                res = httpx.get(f"{API_BASE_URL}/tickets/{lookup_id.strip()}", timeout=3.0)
                if res.status_code == 200:
                    t_data = res.json()
                    st.success(f"Verified: {t_data['ticket_id']}", icon="✅")
                    st.json(t_data)
                elif res.status_code == 404:
                    st.error(f"Ticket '{lookup_id.strip()}' not found.", icon="❌")
                else:
                    st.error(f"HTTP {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please provide a Ticket ID.")


# ==============================================================================
# MAIN VIEW: CHAT INTERFACE & HERO
# ==============================================================================
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">🎧 Customer Support Ticket Agent</h1>
        <div class="app-subtitle">
            Autonomous multi-turn assistant grounded in enterprise policies with automated ticket resolution & voice integration.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# QUICK TEST SCENARIO TOOLBAR (Single clean row, zero vertical clutter)
st.markdown('<div class="starter-header">⚡ Quick Test Scenarios</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.button(
        "📦 Delivery Time",
        key="btn_delivery",
        on_click=queue_prompt,
        args=("How long does standard delivery take?",),
        use_container_width=True,
    )
with c2:
    st.button(
        "🔄 20-Day Return",
        key="btn_returns",
        on_click=queue_prompt,
        args=("Can I return an item after 20 days?",),
        use_container_width=True,
    )
with c3:
    st.button(
        "💳 Payment Issue",
        key="btn_payment",
        on_click=queue_prompt,
        args=("My payment was charged twice for order 1234",),
        use_container_width=True,
    )
with c4:
    st.button(
        "🔐 Password Reset",
        key="btn_password",
        on_click=queue_prompt,
        args=("How do I reset a forgotten password?",),
        use_container_width=True,
    )

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

# 1. RENDER EXISTING CONVERSATION HISTORY
for idx, msg in enumerate(st.session_state.messages):
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    msg_id = msg.get("id", f"msg_{idx}")

    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

        # Render Verified Ticket Creation Card if generated
        ticket_id = msg.get("ticket_id")
        if ticket_id:
            st.markdown(
                f"""
                <div class="ticket-card">
                    <div class="ticket-card-header">
                        <span>🎫 Support Ticket Generated</span>
                        <span class="ticket-badge-open">OPEN</span>
                    </div>
                    <div>Ticket ID: <strong><code>{ticket_id}</code></strong></div>
                    <div style="font-size: 0.8rem; color: #a7f3d0; margin-top: 4px;">
                        A confirmation record has been registered in the system repository. You can inspect this record in the sidebar inspector.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # UNIFIED METADATA & AUDIO ACTION ROW (Sleek, perfectly aligned)
        if msg["role"] == "assistant":
            col_src, col_btn = st.columns([8, 4])

            # Left: Grounded RAG Sources
            with col_src:
                sources = msg.get("sources")
                if sources:
                    badges_html = "".join(f"<span class='source-badge'>📄 {src}</span>" for src in sources)
                    st.markdown(
                        f"""
                        <div class="source-container">
                            <span style="font-size: 0.76rem; color: #71717a; align-self: center;">Sources:</span>
                            {badges_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Right: Listen to Response Button
            with col_btn:
                is_active = st.session_state.active_audio == msg_id
                btn_label = "🔊 Hide Audio" if is_active else "🔊 Listen to Response"
                if st.button(btn_label, key=f"btn_listen_{msg_id}", use_container_width=True):
                    if is_active:
                        st.session_state.active_audio = None
                        st.rerun()
                    else:
                        with st.spinner("Generating voice..."):
                            ok = synthesize_message(msg_id, msg["content"])
                        st.session_state.active_audio = msg_id if ok else None
                        st.rerun()

            # Clean full-width audio player directly below the action row
            if st.session_state.active_audio == msg_id and msg_id in st.session_state.audio_cache:
                st.audio(st.session_state.audio_cache[msg_id], format="audio/mpeg", autoplay=True)
            elif msg_id in st.session_state.tts_errors:
                st.warning(f"🔇 {st.session_state.tts_errors[msg_id]}")


# ==============================================================================
# VOICE INPUT CONTROLS (MICROPHONE + EDITABLE TRANSCRIPT REVIEW)
# ==============================================================================
_mic_open = bool(st.session_state.pending_transcript) or bool(st.session_state.stt_error) \
    or st.session_state.get(f"voice_mic_recorder_{st.session_state.recorder_key}") is not None
with st.expander("🎙️ Dictate Question via Microphone (Speech-to-Text)", expanded=_mic_open):
    st.caption("Record using your microphone, review and edit the transcript, then submit to the support agent.")
    audio_val = st.audio_input("Record voice query", key=f"voice_mic_recorder_{st.session_state.recorder_key}")

    if audio_val is not None:
        audio_bytes = audio_val.getvalue()
        audio_hash = hash(audio_bytes)

        if st.session_state.last_transcribed_audio_hash != audio_hash:
            # Remember this recording first so a failure is not retried on every rerun
            st.session_state.last_transcribed_audio_hash = audio_hash
            st.session_state.stt_error = None
            with st.spinner("Transcribing audio with Whisper STT (the first run can take a minute while the model loads)..."):
                try:
                    files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
                    stt_res = httpx.post(f"{API_BASE_URL}/voice/transcribe", files=files, timeout=180.0)
                    if stt_res.status_code == 200:
                        transcript = stt_res.json().get("transcript", "")
                        st.session_state.pending_transcript = transcript
                        # Push the new text into the editable box (it keeps its old value otherwise)
                        st.session_state["transcript_editor_box"] = transcript
                    else:
                        st.session_state.pending_transcript = None
                        st.session_state.stt_error = get_error_message(
                            stt_res, "No understandable speech was detected"
                        )
                except httpx.ConnectError:
                    st.session_state.stt_error = f"Could not reach the API server at {API_BASE_URL}."
                except Exception as exc:
                    st.session_state.stt_error = f"Transcription error: {exc}"
            st.rerun()

    if st.session_state.stt_error:
        st.warning(f"⚠️ {st.session_state.stt_error} You can record again or use typed chat.")

    # Editable Transcript Review & Confirmation
    if st.session_state.pending_transcript:
        st.markdown("##### 📝 Review & Edit Transcript:")
        user_transcript = st.text_area(
            "Review transcript",
            key="transcript_editor_box",
            height=70,
            label_visibility="collapsed",
        )
        col_c_sub, col_c_disc = st.columns([1, 1])
        with col_c_sub:
            if st.button("🚀 Confirm & Submit to Agent", key="btn_confirm_dictation", use_container_width=True):
                if user_transcript.strip():
                    queue_prompt(user_transcript.strip())
                    st.session_state.pending_transcript = None
                    st.rerun()
        with col_c_disc:
            if st.button("🗑️ Discard Transcript", key="btn_discard_dictation", use_container_width=True):
                st.session_state.pending_transcript = None
                st.rerun()


# ==============================================================================
# 2. CAPTURE ACTIVE INPUT (TYPED INPUT OR QUEUED PROMPT)
# ==============================================================================
active_prompt = None
if prompt := st.chat_input("Ask a policy question or describe an unresolved issue..."):
    active_prompt = prompt
elif st.session_state.next_query:
    active_prompt = st.session_state.next_query
    st.session_state.next_query = None

# ==============================================================================
# 3. PROCESS AND STREAM NEW RESPONSE
# ==============================================================================
if active_prompt:
    clean_prompt = active_prompt.strip()
    user_msg_id = str(uuid.uuid4())[:8]
    asst_msg_id = str(uuid.uuid4())[:8]

    # Append & display user message immediately in chat
    st.session_state.messages.append({"id": user_msg_id, "role": "user", "content": clean_prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(clean_prompt)

    # Display assistant placeholder with immediate loading spinner INSIDE chat bubble
    with st.chat_message("assistant", avatar="🤖"):
        assistant_response = ""
        sources = []
        ticket_id = None
        has_error = False

        with st.spinner("Searching enterprise policies & reasoning..."):
            try:
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": clean_prompt,
                }
                response = httpx.post(f"{API_BASE_URL}/chat", json=payload, timeout=60.0)

                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data.get("response", "")
                    sources = data.get("sources", [])
                    ticket_id = data.get("ticket_id")

                    if ticket_id:
                        st.session_state.latest_ticket = ticket_id
                elif response.status_code == 422:
                    has_error = True
                    assistant_response = "⚠️ Validation error (422): Input message cannot be empty."
                else:
                    has_error = True
                    assistant_response = f"⚠️ API Error (HTTP {response.status_code}): {response.text}"

            except httpx.ConnectError:
                has_error = True
                assistant_response = f"⚠️ Connection Error: Could not connect to API server at {API_BASE_URL}."
            except Exception as exc:
                has_error = True
                assistant_response = f"⚠️ System Error: {exc}"

        # Stream the response word-by-word with typewriter pacing
        if not has_error:
            def stream_words():
                words = assistant_response.split(" ")
                for i, w in enumerate(words):
                    yield w + (" " if i < len(words) - 1 else "")
                    time.sleep(0.012)

            st.write_stream(stream_words())
        else:
            st.error(assistant_response)

        # Save assistant message to session state
        st.session_state.messages.append({
            "id": asst_msg_id,
            "role": "assistant",
            "content": assistant_response,
            "sources": sources,
            "ticket_id": ticket_id,
        })

        # Cleanly rerun to mount in history with instant audio playback
        st.rerun()