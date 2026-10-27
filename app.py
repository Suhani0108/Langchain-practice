"""
Streamlit chat UI for the search agent.

Run:  streamlit run app.py
"""

import uuid

import streamlit as st

from agent_setup import TOOLS, ask, build_agent
from common import MissingAPIKeyError, friendly_error

st.set_page_config(page_title="LangChain Search Agent", page_icon="🔎")
st.title("🔎 LangChain Search Agent")
st.caption("Groq + DuckDuckGo + Wikipedia + calculator + date tool")


# Streamlit re-runs this whole script on every interaction.
# @st.cache_resource builds the agent only once and reuses it.
@st.cache_resource
def get_agent():
    return build_agent()


try:
    agent = get_agent()
except MissingAPIKeyError as error:
    st.error(str(error))
    st.stop()

# st.session_state survives re-runs, so we keep per-user data here:
# - thread_id: which conversation the agent's memory should use
# - chat: the messages to display on screen
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []

with st.sidebar:
    st.subheader("Tools")
    for t in TOOLS:
        st.markdown(f"- `{t.name}`")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat = []
        st.rerun()

# Show the conversation so far.
for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools"):
            st.caption("Tools used: " + ", ".join(message["tools"]))

# Handle a new question.
if question := st.chat_input("Ask me anything..."):
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, tools_used = ask(agent, question, st.session_state.thread_id)
            except Exception as error:
                st.error(friendly_error(error))
                st.stop()
        st.markdown(answer)
        if tools_used:
            st.caption("Tools used: " + ", ".join(tools_used))

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "tools": tools_used}
    )
