import traceback
import streamlit as st


def _extract_module_name(ai_reply: str) -> str | None:
    """
    Very simple heuristic:
    - if reply mentions 'weather', assume weather_panel
    - if reply mentions 'layout', assume layout_manager
    - otherwise None (no auto-update)
    """
    text = ai_reply.lower()
    if "weather" in text:
        return "weather_panel"
    if "layout" in text:
        return "layout_manager"
    return None


def _extract_code_block(ai_reply: str) -> str | None:
    if "```python" not in ai_reply:
        return None
    start = ai_reply.find("```python") + len("```python")
    end = ai_reply.find("```", start)
    if end == -1:
        return None
    return ai_reply[start:end].strip()


def render(
    chat,
    mem_text: str,
    call_jarvis,
    safe_write_module,
    safe_save_json,
    temp_chat_file,
    memory_module,
):
    """
    Renders the chat interface and wires it up to Jarvis.
    """
    # Show history
    for msg in chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask / tell Jarvis something...")

    if not user_msg:
        return

    # Add user message
    chat.append({"role": "user", "content": user_msg})
    safe_save_json(temp_chat_file, chat)

    lower = user_msg.lower().strip()

    # Explicit memory command: "remember ..."
    if lower.startswith("remember "):
        to_store = user_msg[len("remember ") :].strip()
        if to_store:
            memory_module.add_fact(to_store, kind="user")
            ai_reply = f"Got it. I will remember: **{to_store}**"
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
                ai_reply = call_jarvis(chat, mem_text)
                st.markdown(ai_reply)
                chat.append({"role": "assistant", "content": ai_reply})
                safe_save_json(temp_chat_file, chat)

                # Optional: auto-update a module if a python code block is present
                code = _extract_code_block(ai_reply)
                if code:
                    module_name = _extract_module_name(ai_reply)
                    if module_name:
                        ok = safe_write_module(module_name, code)
                        if ok:
                            st.info(
                                f"Module `{module_name}.py` updated. "
                                "Reload the app to see changes."
                            )

            except Exception:
                st.error("Jarvis error.")
                st.code(traceback.format_exc())
