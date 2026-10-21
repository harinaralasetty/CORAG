"""Build our own tiny TrueType test font; no downloaded fonts or document content.

Run from the repository root with fonttools installed. The glyph outlines are
simple rectangles, not a legible typeface. Only the cmap matters for this test.
"""
from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


def main():
    builder = FontBuilder(1000, isTTF=True)
    names = [".notdef", "space", *[f"padding{i}" for i in range(159)],
             "uni0C05", "uni0C06", "uni0C07", "uni0C08"]
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({32: "space", **{cp: f"uni{cp:04X}" for cp in range(0x0C05, 0x0C09)}})
    glyphs = {}
    for i, name in enumerate(names):
        pen = TTGlyphPen(None)
        if name.startswith("uni"):
            pen.moveTo((50, 0))
            pen.lineTo((300 + i, 0))
            pen.lineTo((300 + i, 700))
            pen.lineTo((50, 700))
            pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics(dict.fromkeys(names, (600, 0)))
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "CORAGTeluguFixture", "styleName": "Regular"})
    builder.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    builder.setupPost()
    builder.setupMaxp()
    # Fixed OpenType epoch timestamps make regeneration reproducible.
    builder.font["head"].created = builder.font["head"].modified = 3800000000
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/original-telugu.ttf"
    builder.save(path)
    print(f"Generated {path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
