"""Offline smoke example using original test content and deterministic embeddings."""
import io
import logging
from pathlib import Path
import sys
from unittest.mock import patch

# Allow `python scripts/demo_indic_pdf.py` from a fresh repository checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import config
from fixtures.pdf_factory import broken_telugu_pdf
from preprocessing.pdf_processor import process_pdf
from retrieval import embeddings_indexing as ei


class OfflineEmbeddings:
    """Two fixed dimensions for this smoke test; no model or download needed."""
    def embed_documents(self, texts):
        return [[1.0, 0.0] if 'ఇ' in text else [0.0, 1.0] for text in texts]


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    info = {}
    pdf = broken_telugu_pdf()
    text, _, starts = process_pdf(io.BytesIO(pdf), recovery_info=info)
    if not info:
        raise SystemExit('Recovery unavailable. Install requirements-indic.txt first.')
    assert text == 'అ ఆ ఇ ఈ' and starts == [0, 4, 4]
    print(f'Recovered text: {text}')
    print(f'Page starts: {starts}')
    print(f'Documents: {info["label"]}')
    print('Embedding/search demo uses deterministic offline vectors; no LLM is called.')
    with patch.dict(config.CHUNK_SETTINGS, CHUNK_SIZE=4, CHUNK_OVERLAP=0), \
            patch.object(ei, 'get_embeddings_model', lambda provider: OfflineEmbeddings()):
        index, chunks = ei.generate_embeddings(text, embedding_provider='offline-demo',
                                               source='synthetic-telugu.pdf', page_starts=starts)
        context = ei.fetch_relevant_data([[1.0, 0.0]], index, chunks)
    assert '[Source: synthetic-telugu.pdf, page 3]\nఇ ఈ' in context
    print('Retrieved chunks:')
    print(context)


if __name__ == '__main__':
    main()
