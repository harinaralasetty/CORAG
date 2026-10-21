# Synthetic Indic fixture

`pdf_factory.py` generates PDFs in memory. All text is original test content.
`original-telugu.ttf` is an original font made by
`python scripts/generate_indic_fixture.py`, which requires fontTools. Its four
Telugu glyphs are rectangles; it tests encoded text recovery, not visual quality.
No third-party font or PDF content is included.

The font's Unicode cmap maps U+0C05 through U+0C08 to glyph IDs 161 through 164.
The three-page PDF omits ToUnicode and uses Identity-H. Pages 1 and 3 draw Telugu
vowels separated by a space glyph; page 2 is blank. PyPDF2 returns Latin-1 glyph
codes and a U+0001 space code. Recovery must yield `అ ఆ` and `ఇ ఈ`, and keep
page 3's citation after the blank page.
