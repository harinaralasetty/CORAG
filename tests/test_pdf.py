from preprocessing import pdf_processor


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
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
