"""Offline detection, recovery, failure isolation and citation checks."""
import io
import logging
import sys

import pytest
from PyPDF2 import PdfReader

from preprocessing import pdf_processor
from fixtures.pdf_factory import broken_telugu_pdf, make_pdf


@pytest.mark.parametrize("text,fonts,identity,expected", [
    ("¡\x01¢ £\x01¤", [], False, True),
    ("\ue001\ue002\ue003", [], False, True),
    ("d`fr ns'k", ["ABCDEF+KrutiDev010"], False, True),
    ("glyphs", ["CORAGTeluguFixture"], True, True),
    ("An ordinary English document.", ["Helvetica"], False, False),
    ("English with Identity-H.", ["Helvetica"], True, False),
    ("Café résumé: £12.50, ©2026", ["Helvetica"], False, False),
    ("తెలుగు పుస్తకం", ["NotoSansTelugu"], True, False),
    ("తెలుగు పుస్తకం", ["KrutiDev010"], False, False),
    ("हिन्दी मराठी", ["KrutiDev010"], False, False),
    ("", [], False, False),
])
def test_detection(text, fonts, identity, expected):
    assert pdf_processor.likely_broken_indic(text, fonts, identity) is expected


def test_original_fixture_has_broken_extraction_and_a_blank_page():
    pages = PdfReader(io.BytesIO(broken_telugu_pdf())).pages
    assert [page.extract_text() for page in pages] == ["¡\x01¢", "", "£\x01¤"]


def fail_recovery(*args, **kwargs):
    raise AssertionError("Readable PDFs must not enter optional recovery")


@pytest.mark.parametrize("pdf,expected", [
    (make_pdf([b"English text."]), "English text."),
    (make_pdf([b"\x00\xa1\x00\xa2"], identity=True, font_name="NotoSansTelugu",
              unicode_map={161: "తెలుగు", 162: " పుస్తకం"}), "తెలుగు పుస్తకం"),
], ids=["english", "unicode-telugu"])
def test_readable_pdfs_skip_recovery(monkeypatch, pdf, expected):
    monkeypatch.setattr(pdf_processor, "recover_indic_pages", fail_recovery)
    info = {}
    text, theme, starts = pdf_processor.process_pdf(io.BytesIO(pdf), recovery_info=info)
    assert text == theme == expected
    assert starts == [0]
    assert info == {}


def test_missing_library_keeps_original_text_and_offsets(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "legacylipi", None)
    monkeypatch.setitem(sys.modules, "legacylipi.core.pdf_parser", None)
    info = {}
    with caplog.at_level(logging.INFO):
        text, theme, starts = pdf_processor.process_pdf(io.BytesIO(broken_telugu_pdf()), recovery_info=info)
    assert text == theme == "¡\x01¢ £\x01¤"
    assert starts == [0, 4, 4]
    assert info == {}
    assert "PyPDF2" in caplog.text and "unavailable" in caplog.text


def recovery_api():
    pytest.importorskip("legacylipi.core.pdf_parser")
    pytest.importorskip("fontTools.ttLib")
    pytest.importorskip("uharfbuzz")
    from legacylipi.core.unicode_converter import UnicodeConverter
    return UnicodeConverter


def test_real_recovery_preserves_page_starts_and_citations(monkeypatch, caplog):
    recovery_api()
    import config
    from retrieval import embeddings_indexing as ei

    class OfflineEmbeddings:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for text in texts]

    monkeypatch.setattr(ei, "get_embeddings_model", lambda provider: OfflineEmbeddings())
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_SIZE", 4)
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_OVERLAP", 0)
    info = {}
    with caplog.at_level(logging.INFO):
        text, theme, starts = pdf_processor.process_pdf(io.BytesIO(broken_telugu_pdf()), recovery_info=info)
    assert text == theme == "అ ఆ ఇ ఈ"
    assert starts == [0, 4, 4]
    assert text[starts[2]:] == "ఇ ఈ"
    assert info == {"label": "Telugu text recovered", "methods": ["legacylipi/glyph-id"], "pages": [1, 3]}
    assert "legacylipi/glyph-id" in caplog.text
    _, chunks = ei.generate_embeddings(text, source="synthetic-telugu.pdf", page_starts=starts)
    assert chunks == ["[Source: synthetic-telugu.pdf, page 1]\nఅ ఆ ",
                      "[Source: synthetic-telugu.pdf, page 3]\nఇ ఈ"]


def test_failed_recovery_keeps_original_text(monkeypatch, caplog):
    converter = recovery_api()

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic recovery failure")

    monkeypatch.setattr(converter, "convert_with_recovery", broken)
    info = {}
    text, theme, starts = pdf_processor.process_pdf(io.BytesIO(broken_telugu_pdf()), recovery_info=info)
    assert text == theme == "¡\x01¢ £\x01¤"
    assert starts == [0, 4, 4] and not info
    assert "synthetic recovery failure" in caplog.text


@pytest.mark.parametrize("bad_text", ["", "still mojibake", "అ" + "\ufffd" * 10])
def test_unsuccessful_conversion_is_not_labelled_recovered(monkeypatch, bad_text):
    converter = recovery_api()

    def unchanged(self, document, *args, **kwargs):
        for page in document.pages:
            for block in page.text_blocks:
                block.unicode_text = bad_text
        return document

    monkeypatch.setattr(converter, "convert_with_recovery", unchanged)
    info = {}
    text, _, starts = pdf_processor.process_pdf(io.BytesIO(broken_telugu_pdf()), recovery_info=info)
    assert text == "¡\x01¢ £\x01¤" and starts == [0, 4, 4] and not info


def test_only_broken_pages_are_replaced(monkeypatch):
    converter = recovery_api()
    from legacylipi.core.models import PDFPage, TextBlock

    def converted(self, document, *args, **kwargs):
        # A converter may also change ordinary pages. CORAG must retain them.
        document.pages = [PDFPage(1, [TextBlock("అ ఆ")]), PDFPage(2, [TextBlock("changed")])]
        return document

    monkeypatch.setattr(converter, "convert_with_recovery", converted)
    pdf = make_pdf([b"d`fr", b"English remains readable."], font_name="KrutiDev010")
    # Give page 2 its own normal font rather than an inherited legacy font.
    reader = PdfReader(io.BytesIO(pdf))
    from PyPDF2 import PdfWriter
    from PyPDF2.generic import DictionaryObject, NameObject
    reader.pages[1]["/Resources"]["/Font"][NameObject("/F1")] = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    output = io.BytesIO()
    writer.write(output)
    info = {}
    text, _, starts = pdf_processor.process_pdf(output, recovery_info=info)
    assert text == "అ ఆ\nEnglish remains readable."
    assert starts == [0, 4] and info["pages"] == [1]


def test_real_legacy_devanagari_conversion():
    recovery_api()
    info = {}
    pdf = make_pdf([b"Hkkjr"], font_name="KrutiDev010")
    text, theme, starts = pdf_processor.process_pdf(io.BytesIO(pdf), recovery_info=info)
    assert text == theme == "भारत" and starts == [0]
    assert info == {"label": "Indic text recovered", "methods": ["legacylipi/kruti-dev"], "pages": [1]}


def test_broken_optional_installation_does_not_fail_upload(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "legacylipi.core.pdf_parser":
            raise OSError("native library unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    info = {}
    text, _, starts = pdf_processor.process_pdf(io.BytesIO(broken_telugu_pdf()), recovery_info=info)
    assert text == "¡\x01¢ £\x01¤" and starts == [0, 4, 4] and not info


def test_unused_legacy_font_does_not_reprocess_english(monkeypatch):
    from PyPDF2 import PdfWriter
    from PyPDF2.generic import DictionaryObject, NameObject

    reader = PdfReader(io.BytesIO(make_pdf([b"English stays readable."])))
    reader.pages[0]["/Resources"]["/Font"][NameObject("/Unused")] = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/KrutiDev010"),
    })
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    output = io.BytesIO()
    writer.write(output)
    monkeypatch.setattr(pdf_processor, "recover_indic_pages", fail_recovery)
    text, _, starts = pdf_processor.process_pdf(output)
    assert text == "English stays readable." and starts == [0]


def test_offsets_are_calculated_after_recovery_changes_text_length(monkeypatch):
    monkeypatch.setattr(pdf_processor, "recover_indic_pages", lambda *args: (
        ["అమ్మ", "", "తెలుగు"],
        {"label": "Telugu text recovered", "methods": ["legacylipi/glyph-id"], "pages": [1, 3]},
    ))
    text, _, starts = pdf_processor.process_pdf(io.BytesIO(broken_telugu_pdf()))
    assert text == "అమ్మ తెలుగు" and starts == [0, 5, 5]
    assert text[starts[2]:] == "తెలుగు"


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_recovery_removes_temporary_pdf_and_restores_stream_position(monkeypatch, tmp_path, fail):
    converter = recovery_api()
    from pathlib import Path
    import tempfile
    from preprocessing import indic_pdf

    folders = []
    real_temporary_directory = tempfile.TemporaryDirectory

    def temporary_directory(**kwargs):
        folder = real_temporary_directory(dir=tmp_path, **kwargs)
        folders.append(Path(folder.name))
        return folder

    monkeypatch.setattr(indic_pdf.tempfile, "TemporaryDirectory", temporary_directory)
    if fail:
        def broken(*args, **kwargs):
            raise RuntimeError("test cleanup after failure")
        monkeypatch.setattr(converter, "convert_with_recovery", broken)
    stream = io.BytesIO(broken_telugu_pdf())
    stream.seek(17)
    pages, info = indic_pdf.recover_indic_pages(stream, ["¡\x01¢", "", "£\x01¤"], {1, 3})
    assert stream.tell() == 17
    assert folders and all(not folder.exists() for folder in folders)
    assert bool(info) is not fail
    assert pages == (["¡\x01¢", "", "£\x01¤"] if fail else ["అ ఆ", "", "ఇ ఈ"])
