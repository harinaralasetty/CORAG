"""Generate original, offline PDF fixtures with explicit page and font objects."""
from io import BytesIO
from pathlib import Path

from PyPDF2 import PdfWriter
from PyPDF2.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject,
    create_string_object,
)


def make_pdf(pages, *, font_name="Helvetica", identity=False, font_data=None, unicode_map=None):
    """pages are byte strings of character codes, never text from an external PDF."""
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type0" if identity else "/Type1"),
        NameObject("/BaseFont"): NameObject("/" + font_name),
    })
    if identity:
        descriptor = DictionaryObject({
            NameObject("/Type"): NameObject("/FontDescriptor"),
            NameObject("/FontName"): NameObject("/" + font_name),
            NameObject("/Flags"): NumberObject(4),
            NameObject("/FontBBox"): ArrayObject([NumberObject(v) for v in [0, 0, 600, 800]]),
            **{NameObject(k): NumberObject(v) for k, v in {
                "/ItalicAngle": 0, "/Ascent": 800, "/Descent": -200,
                "/CapHeight": 700, "/StemV": 80,
            }.items()},
        })
        if font_data:
            program = DecodedStreamObject()
            program.set_data(font_data)
            descriptor[NameObject("/FontFile2")] = writer._add_object(program)
        descendant = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/CIDFontType2"),
            NameObject("/BaseFont"): NameObject("/" + font_name),
            NameObject("/CIDToGIDMap"): NameObject("/Identity"),
            NameObject("/DW"): NumberObject(600),
            NameObject("/CIDSystemInfo"): DictionaryObject({
                NameObject("/Registry"): create_string_object("Adobe"),
                NameObject("/Ordering"): create_string_object("Identity"),
                NameObject("/Supplement"): NumberObject(0),
            }),
            NameObject("/FontDescriptor"): writer._add_object(descriptor),
        })
        font[NameObject("/Encoding")] = NameObject("/Identity-H")
        font[NameObject("/DescendantFonts")] = ArrayObject([writer._add_object(descendant)])
    if unicode_map:
        entries = b"\n".join(
            f"<{code:04X}> <{text.encode('utf-16-be').hex()}>".encode()
            for code, text in unicode_map.items()
        )
        cmap = DecodedStreamObject()
        cmap.set_data(
            b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
            b"/CMapName /Fixture def /CMapType 2 def\n"
            b"1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
            + f"{len(unicode_map)} beginbfchar\n".encode() + entries
            + b"\nendbfchar endcmap CMapName currentdict /CMap defineresource pop end end"
        )
        font[NameObject("/ToUnicode")] = writer._add_object(cmap)
    font_ref = writer._add_object(font)
    for codes in pages:
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[-1]
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 14 Tf 72 720 Td <" + codes.hex().encode() + b"> Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def broken_telugu_pdf():
    font_data = Path(__file__).with_name("original-telugu.ttf").read_bytes()
    # Original test text: four independent Telugu vowels. Space GID=1, vowels=161..164.
    return make_pdf([
        bytes.fromhex("00a1000100a2"), b"", bytes.fromhex("00a3000100a4"),
    ], font_name="CORAGTeluguFixture", identity=True, font_data=font_data)
