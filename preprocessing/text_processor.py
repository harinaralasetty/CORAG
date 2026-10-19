from config import THEME_LENGTH


def process_text(content):
    """Plain text or Markdown file bytes: decoded as UTF-8, dropping a byte-order mark and replacing invalid bytes."""
    text = content.decode("utf-8-sig", errors="replace").strip()
    return text, text[:THEME_LENGTH]
