from preprocessing.text_processor import process_text


def test_text_is_decoded_without_bom_and_trimmed():
    text, theme = process_text("﻿  # Title\n\nBody text.\n".encode("utf-8"))
    assert text == "# Title\n\nBody text."
    assert theme == text


def test_invalid_utf8_bytes_are_replaced_not_fatal():
    text, _ = process_text(b"caf\xe9 menu")
    assert text == "caf� menu"


def test_theme_is_capped(monkeypatch):
    import preprocessing.text_processor as tp
    monkeypatch.setattr(tp, "THEME_LENGTH", 5)
    assert tp.process_text(b"abcdefgh") == ("abcdefgh", "abcde")
