import random

from preprocessing import pdf_processor


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self, **kwargs):
        return self.text


def read_pages(monkeypatch, *texts):
    class FakeReader:
        def __init__(self, file):
            self.pages = [FakePage(t) for t in texts]

    monkeypatch.setattr(pdf_processor.PyPDF2, "PdfReader", FakeReader)
    return pdf_processor.process_pdf(None)[0]


def test_words_at_a_page_break_stay_apart(monkeypatch):
    assert read_pages(monkeypatch, "the last word", "first word") == "the last word first word"
    assert read_pages(monkeypatch, "End of page.", "Next page") == "End of page.\nNext page"


def test_cleanup_still_works_across_a_page_break(monkeypatch):
    assert read_pages(monkeypatch, "a hyphen-", "ated ﬁle") == "a hyphenated file"


def page_starts(monkeypatch, *texts):
    class FakeReader:
        def __init__(self, file):
            self.pages = [FakePage(t) for t in texts]

    monkeypatch.setattr(pdf_processor.PyPDF2, "PdfReader", FakeReader)
    text, _, starts = pdf_processor.process_pdf(None)
    return text, starts


def test_page_starts_point_at_each_page_in_the_cleaned_text(monkeypatch):
    text, starts = page_starts(monkeypatch, "a hyphen-", "ated ﬁle", "", "Last page")
    assert text == "a hyphenated file\nLast page"
    assert starts == [0, 8, 18, 18]  # page 3 is empty, so it starts where page 4 does
    assert text[starts[1]:] == "ated file\nLast page" and text[starts[3]:] == "Last page"


def test_offset_cleanup_matches_the_regular_expressions():
    rng = random.Random(7)
    alphabet = ["a", "Z", "-", "\n", "\n", " ", ".", "ﬁ", "ﬃ", "É"]
    for _ in range(20000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        cleaned, where = pdf_processor.clean_with_offsets(raw)
        assert cleaned == pdf_processor.clean_extracted_text_v2(raw), repr(raw)
        assert len(where) == len(raw) + 1 and where[-1] == len(cleaned)
        assert all(0 <= a <= b for a, b in zip(where, where[1:]))
