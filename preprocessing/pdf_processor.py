import PyPDF2 
import re
from config import THEME_LENGTH

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


def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)  # PyPDF2 expects a binary file-like object
    # A newline between pages keeps words apart; the cleanup below turns it into a space or re-joins a hyphenated word
    pages = [page.extract_text() for page in pdf_reader.pages]
    extracted_text, where = clean_with_offsets("\n".join(pages))
    # Where each page begins in the cleaned text, so chunks can cite their page
    page_starts, raw_start = [], 0
    for page in pages:
        page_starts.append(where[raw_start])
        raw_start += len(page) + 1

    # Determine document theme if applicable
    document_theme = extracted_text[:THEME_LENGTH]  # Using the first 2000 characters as a theme
    return extracted_text, document_theme, page_starts
