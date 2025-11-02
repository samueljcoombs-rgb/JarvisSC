import streamlit as st
import openai
import os
import traceback

# --- CONFIG ---
openai.api_key = os.getenv("OPENAI_API_KEY") or "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"

st.set_page_config(page_title="Jarvis Dashboard", layout="wide")

# --- SIMPLE UI ---
st.title("🤖 Jarvis AI Dashboard")
st.write("Talk to your AI assistant below. Type requests like:")
st.write("- *Add a weather widget*")
st.write("- *Change background colour to navy blue*")
st.write("- *Fetch my to-do list from Google Sheets*")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- CHAT INPUT ---
prompt = st.chat_input("Ask Jarvis...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Jarvis thinking..."):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant that manages and edits a Streamlit dashboard called app.py. You have permission to modify this file safely. Only update relevant sections. Return the full new Python code when changes are requested."},
                        *st.session_state.messages
                    ],
                    temperature=0.4,
                )
                reply = response.choices[0].message["content"]
                st.markdown(reply)

                # Detect if AI produced new code
                if "```python" in reply:
                    code_start = reply.find("```python") + 9
                    code_end = reply.find("```", code_start)
                    new_code = reply[code_start:code_end].strip()

                    with open("app.py", "w", encoding="utf-8") as f:
                        f.write(new_code)

                    st.success("✅ Code updated. Streamlit will reload automatically.")
                    st.stop()  # Forces Streamlit reload

                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error("Something went wrong.")
                st.text(traceback.format_exc())
