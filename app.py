# app.py
# Phase 5 — Production Streamlit UI for Yojana Sahayak (Standard RAG AI Chatbot)
# Dark-mode glassmorphic interface, dynamic questionnaire, live match counter, and rich scheme cards.

import streamlit as st
import json
import time
import config
from chatbot import ChatbotSession, QUESTIONS, get_live_count

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Yojana Sahayak | AI Scheme Discovery",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# PREMIUM STYLING — Dark Mode Glassmorphism
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: #070b14 !important;
    color: #e2e8f0;
}
[data-testid="stAppViewContainer"] { background: #070b14 !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
footer, header { display: none !important; }

/* Left Panel Styling */
.brand-header {
    padding: 24px 20px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, transparent 100%);
}
.brand-logo { font-size: 2.2rem; margin-bottom: 4px; display: block; }
.brand-name {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #10b981, #34d399, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-tagline {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* DB Status Pill */
.db-status-wrap { padding: 14px 20px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.db-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    color: #10b981;
}
.db-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
}

/* Progress Tracker */
.progress-section { padding: 16px 20px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.progress-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 10px; }
.progress-bar-track { height: 4px; background: rgba(255,255,255,0.07); border-radius: 4px; overflow: hidden; margin-bottom: 12px; }
.progress-bar-fill { height: 100%; background: linear-gradient(90deg, #10b981, #34d399); border-radius: 4px; transition: width 0.4s ease; }
.step-dots { display: flex; gap: 5px; align-items: center; }
.step-dot { height: 6px; border-radius: 3px; }
.step-dot-done { background: #10b981; width: 16px; }
.step-dot-active { background: #34d399; width: 22px; box-shadow: 0 0 6px #34d399; }
.step-dot-todo { background: rgba(255,255,255,0.1); width: 6px; }

/* Citizen Profile Card */
.profile-section { padding: 16px 20px; }
.section-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-bottom: 12px; }
.profile-grid { display: flex; flex-direction: column; gap: 7px; }
.profile-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.05);
}
.profile-key { font-size: 0.76rem; color: #94a3b8; font-weight: 500; }
.profile-val-filled { font-size: 0.8rem; color: #10b981; font-weight: 700; background: rgba(16,185,129,0.1); padding: 2px 7px; border-radius: 4px; }
.profile-val-empty { font-size: 0.75rem; color: #475569; font-style: italic; }

/* Match Counter Badge */
.match-counter {
    margin: 12px 20px;
    padding: 14px 18px;
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(52,211,153,0.04));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 12px;
    text-align: center;
}
.match-number { font-size: 2.2rem; font-weight: 800; color: #10b981; line-height: 1; }
.match-label { font-size: 0.7rem; color: #64748b; margin-top: 4px; letter-spacing: 0.5px; font-weight: 600; }

/* Right Panel — Chat Interface */
.chat-topbar {
    padding: 16px 28px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(13,17,23,0.85);
    backdrop-filter: blur(12px);
}
.chat-topbar-title { font-size: 1.05rem; font-weight: 700; color: #f8fafc; }
.chat-topbar-sub { font-size: 0.76rem; color: #64748b; margin-top: 2px; }

/* Chat Messages */
.chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 18px;
}
.msg-row { display: flex; gap: 12px; align-items: flex-start; }
.msg-row-user { flex-direction: row-reverse; }

.avatar {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.05rem; flex-shrink: 0;
}
.avatar-bot { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid rgba(16,185,129,0.4); }
.avatar-user { background: linear-gradient(135deg, #1e3a5f, #1e40af); border: 1px solid rgba(59,130,246,0.4); }

.bubble {
    max-width: 75%;
    padding: 13px 17px;
    border-radius: 14px;
    font-size: 0.88rem;
    line-height: 1.6;
}
.bubble-bot { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-top-left-radius: 4px; color: #e2e8f0; }
.bubble-user { background: linear-gradient(135deg, #1d4ed8, #1e40af); border: 1px solid rgba(59,130,246,0.3); border-top-right-radius: 4px; color: #eff6ff; }

/* Scheme Recommendation Cards */
.rec-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.06), rgba(5,150,105,0.03));
    border: 1px solid rgba(16,185,129,0.22);
    border-radius: 12px;
    padding: 16px 18px;
    margin-top: 10px;
    position: relative;
}
.rec-card-rank {
    position: absolute; top: 14px; right: 14px;
    font-size: 0.65rem; font-weight: 700; color: #10b981;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
    padding: 2px 7px; border-radius: 14px;
}
.rec-card-name { font-size: 0.95rem; font-weight: 700; color: #f0fdf4; margin-bottom: 8px; padding-right: 60px; }
.rec-card-row { display: flex; gap: 6px; font-size: 0.82rem; margin-bottom: 5px; }
.rec-card-icon { flex-shrink: 0; }
.rec-card-text { color: #94a3b8; }
.rec-card-text strong { color: #e2e8f0; }
.rec-card-link {
    display: inline-block; margin-top: 8px; padding: 5px 12px;
    background: linear-gradient(135deg, #10b981, #059669); color: #fff !important;
    font-size: 0.76rem; font-weight: 700; border-radius: 6px; text-decoration: none;
}

/* Form inputs & buttons */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #e2e8f0 !important; padding: 10px 14px !important;
}
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: white !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# Session State Init
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatbotSession()

chatbot = st.session_state.chatbot
prof    = chatbot.profile

# Scheme HTML renderer helper
def render_scheme_cards(content: str) -> str:
    if "Scheme Name:" not in content:
        return f"<p style='color:#94a3b8;font-size:.88rem;line-height:1.6'>{content.replace(chr(10),'<br>')}</p>"

    parts = content.split("Scheme Name:")
    prefix = parts[0].strip()
    cards_html = ""

    if prefix:
        cards_html += f"<p style='color:#94a3b8;font-size:.88rem;margin-bottom:12px'>{prefix.replace(chr(10),'<br>')}</p>"

    icons = {"Why you qualify": "✅", "Benefit": "💰", "How to apply": "📋", "Link": "🔗"}

    for idx, part in enumerate(parts[1:], 1):
        lines = part.strip().splitlines()
        name  = lines[0].strip() if lines else f"Scheme {idx}"
        body_html = ""
        link_html = ""

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            matched = False
            for key, icon in icons.items():
                if line.startswith(key + ":"):
                    val = line[len(key)+1:].strip()
                    if key == "Link":
                        link_html = f'<a class="rec-card-link" href="{val}" target="_blank">🔗 Official Website ↗</a>'
                    else:
                        body_html += f'<div class="rec-card-row"><span class="rec-card-icon">{icon}</span><span class="rec-card-text"><strong>{key}:</strong> {val}</span></div>'
                    matched = True
                    break
            if not matched and line:
                body_html += f'<div class="rec-card-row"><span class="rec-card-text">{line}</span></div>'

        cards_html += f"""
        <div class="rec-card">
            <div class="rec-card-rank">#{idx} MATCH</div>
            <div class="rec-card-name">{name}</div>
            {body_html}
            {link_html}
        </div>"""

    return cards_html

PROFILE_FIELDS = [
    ("📍 State/UT",   "state"),
    ("🎂 Age",        "age"),
    ("⚥  Gender",     "gender"),
    ("💼 Occupation", "occupation"),
    ("💵 Income",     "income"),
    ("🏷️ Category",   "category"),
    ("🎯 Interest",   "interest"),
]

def profile_row(label: str, val) -> str:
    if val is not None and val != "":
        return f'<div class="profile-row"><span class="profile-key">{label}</span><span class="profile-val-filled">{val}</span></div>'
    return f'<div class="profile-row"><span class="profile-key">{label}</span><span class="profile-val-empty">—</span></div>'

# Two-column layout: Left Dashboard | Right Chat
left_col, right_col = st.columns([1, 2.5], gap="small")

# ════════════════════════════════════════════
# LEFT PANEL — Citizen Profile & Dashboard
# ════════════════════════════════════════════
with left_col:
    st.markdown("""
    <div class="brand-header">
        <span class="brand-logo">🏛️</span>
        <div class="brand-name">Yojana Sahayak</div>
        <div class="brand-tagline">Standard RAG · AI Scheme Discovery</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="db-status-wrap">
        <div class="db-pill"><div class="db-dot"></div>ChromaDB Vector Store Active</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress Bar
    total_steps = len(QUESTIONS)
    done_steps  = chatbot.step
    pct         = int((done_steps / total_steps) * 100)
    dots_html   = ""
    for i in range(total_steps):
        if i < done_steps:
            dots_html += '<div class="step-dot step-dot-done"></div>'
        elif i == done_steps:
            dots_html += '<div class="step-dot step-dot-active"></div>'
        else:
            dots_html += '<div class="step-dot step-dot-todo"></div>'

    st.markdown(f"""
    <div class="progress-section">
        <div class="progress-label">Profile Progress — {pct}%</div>
        <div class="progress-bar-track">
            <div class="progress-bar-fill" style="width:{pct}%"></div>
        </div>
        <div class="step-dots">{dots_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Citizen Profile
    rows_html = "".join(profile_row(label, prof.get(key)) for label, key in PROFILE_FIELDS)
    st.markdown(f"""
    <div class="profile-section">
        <div class="section-title">👤 Citizen Profile</div>
        <div class="profile-grid">{rows_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Live Match Counter Badge
    match_count = get_live_count(prof)
    st.markdown(f"""
    <div class="match-counter">
        <div class="match-number">{match_count}</div>
        <div class="match-label">MATCHING SCHEMES FOUND</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🧹 Reset Conversation", key="btn_reset"):
        st.session_state.chatbot = ChatbotSession()
        st.rerun()

# ════════════════════════════════════════════
# RIGHT PANEL — Chat Interface
# ════════════════════════════════════════════
with right_col:
    step_label = f"Question {min(chatbot.step + 1, len(QUESTIONS))} of {len(QUESTIONS)}" if chatbot.step < len(QUESTIONS) else "Profile Complete ✓"
    st.markdown(f"""
    <div class="chat-topbar">
        <div class="chat-topbar-title">💬 Conversation with Yojana Sahayak</div>
        <div class="chat-topbar-sub">Personalized AI Guide for Indian Welfare Schemes • {step_label}</div>
    </div>
    """, unsafe_allow_html=True)

    # Initial Welcome message
    if not chatbot.history:
        chatbot.history.append({
            "role": "assistant",
            "content": QUESTIONS[0]["question"]
        })

    # Chat Messages
    chat_html = '<div class="chat-messages" id="chat-box">'
    for msg in chatbot.history:
        role    = msg["role"]
        content = msg["content"]
        if role == "assistant":
            rendered = render_scheme_cards(content)
            chat_html += f"""
            <div class="msg-row">
                <div class="avatar avatar-bot">🏛️</div>
                <div class="bubble bubble-bot">{rendered}</div>
            </div>"""
        else:
            safe = content.replace("<", "&lt;").replace(">", "&gt;")
            chat_html += f"""
            <div class="msg-row msg-row-user">
                <div class="avatar avatar-user">👤</div>
                <div class="bubble bubble-user">{safe}</div>
            </div>"""
    chat_html += '</div>'

    st.markdown(chat_html, unsafe_allow_html=True)

    # Quick Select Options
    if chatbot.step < len(QUESTIONS):
        opts = QUESTIONS[chatbot.step]["options"]
        if opts:
            st.markdown("<div style='font-size:0.75rem;color:#64748b;margin:10px 0 6px;font-weight:600'>QUICK SELECT:</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(opts), 4))
            for i, opt in enumerate(opts):
                with cols[i % len(cols)]:
                    if st.button(opt, key=f"opt_{chatbot.step}_{i}"):
                        chatbot.process_user_message(opt)
                        st.rerun()

    # Form text input
    placeholder = "Type your response..." if chatbot.step < len(QUESTIONS) else "Ask any follow-up scheme question..."
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            user_text = st.text_input("", placeholder=placeholder, label_visibility="collapsed", key="chat_input")
        with col_btn:
            submitted = st.form_submit_button("Send ➔")

        if submitted and user_text.strip():
            chatbot.process_user_message(user_text.strip())
            st.rerun()
