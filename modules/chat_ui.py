import streamlit as st
import traceback


def render(
    chat,
    mem_text,
    call_jarvis,
    safe_write_module,
    safe_save_json,
    temp_chat_file,
    memory_module,
):
    """
    Main chat interface — shows chat, accepts input, handles Jarvis responses.
    """

    st.header("💬 Chat with Jarvis")

    # Show history
    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # One and only chat_input on the page
    user_msg = st.chat_input("Message Jarvis...", key="main_chat_input")

    if not user_msg:
        return

    # Append user message
    chat.append({"role": "user", "content": user_msg})
    safe_save_json(temp_chat_file, chat)
    lower = user_msg.lower().strip()

    # Explicit memory command
    if lower.startswith("remember "):
        to_store = user_msg[len("remember "):].strip()
        if to_store:
            memory_module.add_fact(to_store, kind="user")
            ai_reply = f"Got it — I’ll remember: **{to_store}**"
        else:
            ai_reply = "You said 'remember' but didn’t tell me what to remember."

        chat.append({"role": "assistant", "content": ai_reply})
        safe_save_json(temp_chat_file, chat)
        with st.chat_message("assistant"):
            st.markdown(ai_reply)
        return

    # Normal AI flow
    with st.chat_message("assistant"):
        with st.spinner("Jarvis thinking..."):
            try:
                ai_reply = call_jarvis(chat, memory_module.recent_summary())
                st.markdown(ai_reply)
                chat.append({"role": "assistant", "content": ai_reply})
                safe_save_json(temp_chat_file, chat)

                # Self-update: if Jarvis outputs full module code
                if "```python" in ai_reply:
                    start = ai_reply.find("```python") + len("```python")
                    end = ai_reply.find("```", start)
                    if end != -1:
                        code = ai_reply[start:end].strip()

                        # Naive module detection: user should mention the module name in the code
                        for target_module in ["chat_ui", "weather_panel", "layout_manager"]:
                            if target_module in code:
                                if safe_write_module(target_module, code):
                                    st.success(f"Updated {target_module}.py — reloading...")
                                    st.stop()

            except Exception:
                st.error("Jarvis encountered an error.")
                st.code(traceback.format_exc())
