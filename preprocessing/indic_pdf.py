"""Optional indic-legacy-pdf adapter. Import and conversion failures retain PyPDF2 text."""
import logging
from pathlib import Path
import tempfile


log = logging.getLogger(__name__)


def indic_count(text):
    return sum(0x0900 <= ord(char) <= 0x0DFF or 0xA8E0 <= ord(char) <= 0xA8FF
               for char in text)


def recover_indic_pages(stream, original_pages, selected):
    """Return (pages, recovery metadata), accepting only improved selected pages.

    legacylipi needs a real path for embedded font discovery. The temporary PDF
    is private to this upload and removed on success or failure. No translation,
    OCR, subprocesses or network services are used.
    """
    try:
        from legacylipi.core.pdf_parser import PDFParser
        from legacylipi.core.encoding_detector import EncodingDetector
        from legacylipi.core.unicode_converter import UnicodeConverter
    except ImportError:
        log.info('Indic recovery unavailable; using PyPDF2 (install requirements-indic.txt)')
        return original_pages, {}
    except Exception:
        log.warning('Indic recovery import failed; using PyPDF2', exc_info=True)
        return original_pages, {}

    try:
        position = stream.tell()
        try:
            stream.seek(0)
            content = stream.read()
        finally:
            stream.seek(position)
        with tempfile.TemporaryDirectory(prefix='corag_indic_') as folder:
            path = Path(folder) / 'input.pdf'
            path.write_bytes(content)
            with PDFParser(path) as parser:
                document = parser.parse()
            overall, encodings = EncodingDetector().detect_from_document(document)
            selected_encodings = {n: encoding for n, encoding in encodings.items() if n in selected}
            converted = UnicodeConverter().convert_with_recovery(document, overall, selected_encodings)
            if len(converted.pages) != len(original_pages):
                raise ValueError('Indic recovery returned a different page count')
            pages, recovered, methods = list(original_pages), [], set()
            for page in converted.pages:
                number = page.page_number
                if number not in selected or number not in selected_encodings:
                    continue
                text = page.unicode_text
                # A heuristic match alone does not establish successful recovery.
                if (not text.strip() or indic_count(text) <= indic_count(pages[number - 1])
                        or text.count('\ufffd') / len(text) > 0.05):
                    continue
                pages[number - 1] = text
                recovered.append(number)
                method = 'legacylipi/' + selected_encodings[number].detected_encoding
                methods.add(method)
                log.info('PDF page %s extracted with %s', number, method)
            for warning in converted.warnings:
                log.warning('Indic recovery warning: %s', warning)
            if recovered:
                telugu = any('\u0c00' <= char <= '\u0c7f' for n in recovered for char in pages[n - 1])
                return pages, {
                    'label': 'Telugu text recovered' if telugu else 'Indic text recovered',
                    'methods': sorted(methods),
                    'pages': sorted(recovered),
                }
    except Exception:
        # This dependency is optional: even incomplete installations or malformed
        # font programs must not turn an otherwise readable upload into an error.
        log.warning('Indic recovery failed; using PyPDF2', exc_info=True)
    return original_pages, {}
