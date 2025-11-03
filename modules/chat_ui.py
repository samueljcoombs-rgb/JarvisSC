import streamlit as st
import traceback


def render(chat, mem_text, call_jarvis, safe_write_module, safe_save_json, temp_chat_file, memory_module):
    """
    Main chat interface — shows chat, accepts input, handles Jarvis responses.
    """

    st.header("💬 Chat with Jarvis")

    # Display history
    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Unique key per session avoids duplicate widget errors
    user_msg = st.chat_input(
        "Message Jarvis...",
        key=f"chat_input_{id(st.session_state)}"
    )

    if user_msg:
        chat.append({"role": "user", "content": user_msg})
        safe_save_json(temp_chat_file, chat)
        lower = user_msg.lower().strip()

        # Handle "remember" command
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

        else:
            # Normal AI reply
            with st.chat_message("assistant"):
                with st.spinner("Jarvis thinking..."):
                    try:
                        ai_reply = call_jarvis(chat, memory_module.recent_summary())
                        st.markdown(ai_reply)
                        chat.append({"role": "assistant", "content": ai_reply})
                        safe_save_json(temp_chat_file, chat)

                        # Check if Jarvis proposed module code updates
                        if "```python" in ai_reply:
                            start = ai_reply.find("```python") + len("```python")
                            end = ai_reply.find("```", start)
                            if end != -1:
                                code = ai_reply[start:end].strip()

                                # Identify target module and update safely
                                for target_module in ["chat_ui", "weather_panel", "layout_manager"]:
                                    if target_module in code:
                                        safe_write_module(target_module, code)
                                        st.stop()

                    except Exception:
                        st.error("Jarvis encountered an error.")
                        st.code(traceback.format_exc())
