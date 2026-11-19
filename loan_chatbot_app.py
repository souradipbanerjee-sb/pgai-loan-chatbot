"""
Web chat UI (Streamlit) on top of the same engine.

Usage:  streamlit run loan_chatbot_app.py
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dotenv import load_dotenv
load_dotenv(os.path.join(HERE, ".env"))          # local runs
import streamlit as st

# Streamlit Community Cloud: the key comes from the app's Secrets settings (never from the repo)
try:
    for _k in ("GROQ_API_KEY", "GROQ_MODEL", "SHEETS_WEBHOOK_URL", "SHEETS_TOKEN"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:                                 # no secrets file locally -> .env is used
    pass

from chatbot_engine import LoanChatbot           # import after the key is set

st.set_page_config(page_title="PGAI two-wheeler loan assistant", page_icon=":motorcycle:")
st.title("Two-wheeler loan assistant")
st.caption("Instant eligibility check · about 2 minutes · no name or contact details needed")

if "bot" not in st.session_state or st.sidebar.button("Start a new application"):
    st.session_state.bot = LoanChatbot(channel="web")
    st.session_state.history = [("assistant", st.session_state.bot.start())]

bot = st.session_state.bot
for role, text in st.session_state.history:
    st.chat_message(role).write(text.replace("\n", "  \n"))

if not bot.done:
    if prompt := st.chat_input("Type your answer"):
        st.session_state.history.append(("user", prompt))
        st.session_state.history.append(("assistant", bot.handle(prompt)))
        st.rerun()
else:
    st.info("Application complete. Use 'Start a new application' in the sidebar to begin again.")

with st.sidebar:
    st.subheader("Progress")
    st.progress(min(len(bot.app) / 6, 1.0), text=f"{len(bot.app)} of 6 details captured")
    st.write(f"Time elapsed: {bot.elapsed_s:.0f} s (target 3 min)")
    with st.expander("Operator view (not shown to customers)"):
        st.write(f"LLM: {'Groq ' + bot.llm.model if bot.llm else 'offline'}")
        st.write(f"Session: `{bot.session_id[:12]}`")
        sink = bot.repo.sheets
        st.write("Google Sheets log: " + (f"on, {sink.sent} rows sent, {sink.failed} failed"
                                          + (f" (last error: {sink.last_error})" if sink.failed else "")
                                          if sink else "off (SHEETS_WEBHOOK_URL not set)"))
        for d in bot.decisions:
            st.json({k: d[k] for k in ("reference", "probability_decline", "threshold", "decision",
                                       "latency_ms", "decision_latency_ms", "features")})
