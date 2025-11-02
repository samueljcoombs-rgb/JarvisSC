import os
import traceback
import streamlit as st

# we use the new openai client (openai>=1.x)
try:
    from openai import OpenAI
except ImportError:
    raise RuntimeError("You need to install openai: pip install openai")

import memory  # <- your memory.py in the repo


# ----------------- CONFIG -----------------
st.set_page_config(page_title="Jarvis AI Dashboard", layout="wide")

# 1) get OpenAI key from env or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # try streamlit secrets
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.error("No OPENAI_API_KEY found. Add it in Streamlit secrets or env.")
    st.stop()

src = "env" if os.getenv("OPENAI_API_KEY") else "st.secrets"
st.sidebar.write(f"🔍 Loaded from {src}: {OPENAI_API_KEY[:10]}")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------- HELPER: call OpenAI -----------------
def call_jarvis(chat_history, mem_text: str):
    """
    chat_history: list of {"role": "...", "content": "..."} from st.session_state
    mem_text: recent memory from memory.py
    returns: assistant text
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are Jarvis, an AI assistant living inside a Streamlit app called app.py.\n"
            "You MAY be asked to update the code of app.py. When (and only when) the user asks "
            "for a code change, respond with FULL Python code for the ENTIRE app.py inside a "
            "```python ... ``` block.\n"
            "You also have ACCESS to a persistent memory.\n"
            "Here is what you currently remember:\n"
            f"{mem_text}\n"
            "If the user says 'remember ...', summarise and save it.\n"
            "If the user just chats, answer normally."
        )
    }

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_msg] + chat_history,
        temperature=0.4,
    )
    return resp.choices[0].message.content


# ----------------- SESSION STATE -----------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_ai" not in st.session_state:
    st.session_state.last_ai = ""

# ----------------- SIDEBAR (memory) -----------------
with st.sidebar:
    st.header("🧠 Memory")
    # show memory
    mem_text = memory.recent_summary()
    if mem_text:
        st.write(mem_text)
    else:
        st.write("No memories yet.")

    # add new memory manually
    new_mem = st.text_input("Add to memory (manual):", key="sidebar_mem_input")
    if new_mem:
        memory.add_fact(new_mem, kind="manual")
        st.success("Saved to memory.")
        # refresh sidebar
        st.rerun()


# ----------------- MAIN UI -----------------
st.title("🤖 Jarvis AI Dashboard (with memory)")

st.write("Talk to Jarvis below. You can say things like:")
st.write("- **remember** that I prefer evening workouts")
st.write("- **what do you remember about me?**")
st.write("- **change the layout** to put weather at the top (Jarvis will rewrite app.py)")

st.divider()

# show chat history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
user_msg = st.chat_input("Ask / tell Jarvis something...")
if user_msg:
    # 1) add user msg to chat
    st.session_state.chat.append({"role": "user", "content": user_msg})

    # 2) special case: user explicitly wants to store memory
    # e.g. "remember I live in Basingstoke"
    lower = user_msg.lower().strip()
    if lower.startswith("remember "):
        # just store the rest
        to_store = user_msg[len("remember "):].strip()
        if to_store:
            memory.add_fact(to_store, kind="user")
            ai_reply = f"Got it. I will remember: **{to_store}**"
            st.session_state.chat.append({"role": "assistant", "content": ai_reply})
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
        else:
            ai_reply = "You said 'remember' but didn't tell me what to remember."
            st.session_state.chat.append({"role": "assistant", "content": ai_reply})
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
    else:
        # 3) normal AI flow
        with st.chat_message("assistant"):
            with st.spinner("Jarvis thinking..."):
                try:
                    mem_now = memory.recent_summary()
                    ai_reply = call_jarvis(st.session_state.chat, mem_now)

                    # show reply
                    st.markdown(ai_reply)
                    st.session_state.chat.append({"role": "assistant", "content": ai_reply})

                    # 4) did Jarvis output new code?
                    if "```python" in ai_reply:
                        start = ai_reply.find("```python") + len("```python")
                        end = ai_reply.find("```", start)
                        if end != -1:
                            new_code = ai_reply[start:end].strip()
                            # write to app.py
                            with open("app.py", "w", encoding="utf-8") as f:
                                f.write(new_code)
                            st.success("✅ Code updated — rerunning app...")
                            st.stop()  # force Streamlit to reload
                except Exception as e:
                    st.error("Jarvis error.")
                    st.code(traceback.format_exc())
