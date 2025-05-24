# logic_handler.py
import html
from typing import Tuple



def prioritize_reply_logic(search_text: str, chat_text: str) -> Tuple[str, str]:
    """
    Combines structured product results and LLM-generated chat into a final Telegram reply.

    Returns:
        final_text (str): Combined, trimmed message to send to user.
        reason (str): Description of which fallback logic applied (for logging/debug).
    """
    search_text = search_text.strip()
    chat_text = chat_text.strip()

    if search_text.startswith("\u274C No matching products found."):
        if chat_text:
            return chat_text, "AI insight only (no matches)"
        else:
            return search_text, "Search failed, no AI insight"
    elif search_text and chat_text:
        combined = f"{search_text}\n\n<b>AI Insight:</b>\n{chat_text}"
        return combined, "Combined search + AI insight"
    elif search_text:
        return search_text, "Only product match"
    elif chat_text:
        return chat_text, "Only AI insight"
    return "\u274C No relevant products or insights found.", "Complete fallback"


# In main_server.py — import this function and use it inside handle_message:
# from logic_handler import prioritize_reply_logic
# ...
# reply_text, reason = prioritize_reply_logic(search_text, chat_text)
# logger.info(f"📤 Reply strategy: {reason}")
