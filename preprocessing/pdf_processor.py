import logging
import re

import PyPDF2
from config import THEME_LENGTH
from preprocessing.indic_pdf import indic_count, recover_indic_pages


log = logging.getLogger(__name__)

# Subset prefixes and punctuation vary between producers.
LEGACY_FONT = re.compile(
    r"shree[-_ ]?(?:dev|lipi)|sdl[-_ ]?dev|dvb[w]?tt|kruti[-_ ]?dev|"
    r"chanakya|aps[-_ ]?(?:c[-_ ]?)?dv|shusha", re.I,
)
TELUGU_FONT = re.compile(r"telugu|gautami|pothana|vemana", re.I)


def likely_broken_indic(text, font_names=(), identity_without_unicode=False):
    """Conservative page-level gate, before importing the optional converter.

    Readable Indic pages take precedence over font names. A few accented Latin
    characters, currency symbols or copyright marks are normal English text.
    Identity-H alone is also common in ordinary, readable PDFs.
    """
    chars = [char for char in text if not char.isspace()]
    if chars and indic_count(text) / len(chars) >= 0.02:
        return False
    if any(LEGACY_FONT.search(name) for name in font_names) and chars:
        return True
    if identity_without_unicode and any(TELUGU_FONT.search(name) for name in font_names):
        return True
    if not chars:
        return False
    latin = sum(0x80 <= ord(char) <= 0xFF for char in chars)
    private = sum(0xE000 <= ord(char) <= 0xF8FF or 0xF0000 <= ord(char) <= 0x10FFFD
                  for char in chars)
    controls = sum(ord(char) < 32 or char == "\ufffd" for char in chars)
    return ((latin >= 2 and latin / len(chars) >= 0.3)
            or (private >= 2 and private / len(chars) >= 0.15)
            or (identity_without_unicode and controls >= 2 and controls / len(chars) >= 0.15))


def _extract_page_with_fonts(page):
    """Collect fonts from text visited by PyPDF2, including Form text.

    Unused font resources must not turn an ordinary English page into an Indic
    candidate. The visitor observes extraction without changing its output.
    """
    names, identity = set(), False

    def remember_font(text, cm, tm, font, size):
        nonlocal identity
        if text.strip() and font is not None:
            names.add(str(font.get("/BaseFont", "")))
            identity |= font.get("/Encoding") in ("/Identity-H", "/Identity-V") and "/ToUnicode" not in font

    text = page.extract_text(visitor_text=remember_font) or ""
    return text, names, identity


def clean_extracted_text_v2(text):
    ligature_map = {
        "ﬁ": "fi",
        "ﬀ": "ff",
        "ﬂ": "fl",
        "ﬃ": "ffi",
    }
    for ligature, replacement in ligature_map.items():
        text = text.replace(ligature, replacement)

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n|[A-Z])', ' ', text)
    
    return text

def clean_with_offsets(text):
    """clean_extracted_text_v2 that also maps offsets: returns (cleaned, where).

    where[i] is the position in cleaned of raw character i, or of the next kept character when i was
    removed; where[len(text)] is len(cleaned). The steps mirror the regular expressions above.
    """
    ligature_map = {"ﬁ": "fi", "ﬀ": "ff", "ﬂ": "fl", "ﬃ": "ffi"}
    where = list(range(len(text) + 1))

    def step(text, keep):
        # keep(text, i) -> (output for text[i], how many raw characters it consumes)
        out, moved, i = [], [], 0
        while i < len(text):
            piece, used = keep(text, i)
            moved.extend([len(out)] * used)
            out.append(piece)
            i += used
        moved.append(len(out))
        flat, offsets = "".join(out), [0]
        for piece in out:
            offsets.append(offsets[-1] + len(piece))
        return flat, [offsets[m] for m in moved]

    def apply(keep):
        nonlocal text, where
        text, moved = step(text, keep)
        where = [moved[w] for w in where]

    apply(lambda t, i: (ligature_map.get(t[i], t[i]), 1))
    apply(lambda t, i: ("", 1) if t[i] == "\n" and i > 0 and t[i - 1] == "\n" else (t[i], 1))
    apply(lambda t, i: ("", 2) if t.startswith("-\n", i) else (t[i], 1))
    apply(lambda t, i: (" ", 1) if t[i] == "\n" and (i + 1 == len(t) or (t[i + 1] != "\n" and not "A" <= t[i + 1] <= "Z"))
          else (t[i], 1))
    return text, where


def process_pdf(file, *, recovery_info=None):
    """Return cleaned text, theme and page starts; optionally fill recovery_info.

    recovery_info belongs to this upload, so simultaneous uploads never share
    recovery status. Existing callers keep the original three-value contract.
    """
    pdf_reader = PyPDF2.PdfReader(file)  # PyPDF2 expects a binary file-like object
    # A newline between pages keeps words apart; the cleanup below turns it into a space or re-joins a hyphenated word
    pages, selected = [], set()
    for number, page in enumerate(pdf_reader.pages, 1):
        text, fonts, identity = _extract_page_with_fonts(page)
        pages.append(text)
        if likely_broken_indic(text, fonts, identity):
            selected.add(number)
    info = {}
    if selected:
        pages, info = recover_indic_pages(pdf_reader.stream, pages, selected)
    if recovery_info is not None:
        recovery_info.update(info)
    log.info('PDF extraction methods: %s', ', '.join(['PyPDF2', *info.get('methods', [])]))
    extracted_text, where = clean_with_offsets("\n".join(pages))
    # Where each page begins in the cleaned text, so chunks can cite their page
    page_starts, raw_start = [], 0
    for page in pages:
        page_starts.append(where[raw_start])
        raw_start += len(page) + 1

    # Determine document theme if applicable
    document_theme = extracted_text[:THEME_LENGTH]  # Using the first 2000 characters as a theme
    return extracted_text, document_theme, page_starts
