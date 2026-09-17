"""
history.py
Handles saving and loading chat sessions to/from local JSON files.
Each session is stored as: chat_history/{doc_name}_{YYYY-MM-DD_HH-MM}.json
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Directory where chat sessions are persisted
HISTORY_DIR = Path(__file__).parent.parent / "chat_history"


def ensure_dir():
    """Create the chat_history directory if it doesn't exist."""
    HISTORY_DIR.mkdir(exist_ok=True)


def _safe_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()


def save_session(document_name: str, chat_history: list, session_id: str = None) -> str:
    """
    Save the current chat session to a JSON file.

    Args:
        document_name: Name of the PDF document.
        chat_history: List of message dicts from st.session_state.
        session_id: Optional existing session ID to overwrite. If None, creates new.

    Returns:
        The session_id (filename without extension) used.
    """
    ensure_dir()

    if session_id is None:
        # New session — generate a unique ID from doc name + timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        safe_doc  = _safe_filename(os.path.splitext(document_name)[0])[:30]
        session_id = f"{safe_doc}_{timestamp}"

    filepath = HISTORY_DIR / f"{session_id}.json"

    data = {
        "session_id":    session_id,
        "document_name": document_name,
        "saved_at":      datetime.now().isoformat(),
        "message_count": sum(1 for m in chat_history if m["role"] == "user"),
        "messages":      chat_history,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return session_id


def load_session(session_id: str) -> dict:
    """
    Load a saved chat session by its ID.

    Returns:
        Dict with keys: session_id, document_name, saved_at, messages
    """
    filepath = HISTORY_DIR / f"{session_id}.json"
    if not filepath.exists():
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[dict]:
    """
    List all saved sessions, sorted by most recent first.

    Returns:
        List of dicts: [{session_id, document_name, saved_at, message_count}]
    """
    ensure_dir()
    sessions = []

    for filepath in sorted(HISTORY_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "session_id":    data.get("session_id", filepath.stem),
                "document_name": data.get("document_name", "Unknown"),
                "saved_at":      data.get("saved_at", ""),
                "message_count": data.get("message_count", 0),
            })
        except Exception:
            continue  # Skip corrupted files

    return sessions


def delete_session(session_id: str):
    """Delete a saved session file."""
    filepath = HISTORY_DIR / f"{session_id}.json"
    if filepath.exists():
        filepath.unlink()


def format_saved_at(iso_str: str) -> str:
    """Format ISO datetime to a readable label like 'Today 3:42 PM'."""
    try:
        dt    = datetime.fromisoformat(iso_str)
        today = datetime.now().date()
        if dt.date() == today:
            return f"Today {dt.strftime('%I:%M %p')}"
        elif (today - dt.date()).days == 1:
            return f"Yesterday {dt.strftime('%I:%M %p')}"
        else:
            return dt.strftime("%b %d, %I:%M %p")
    except Exception:
        return iso_str

