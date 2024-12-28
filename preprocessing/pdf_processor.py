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

def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)  # PyPDF2 expects a binary file-like object
    extracted_text = ""
    for page in pdf_reader.pages:  # Access the pages directly
        extracted_text += page.extract_text()

    extracted_text = clean_extracted_text_v2(extracted_text)
    # Determine document theme if applicable
    document_theme = extracted_text[:THEME_LENGTH]  # Using the first 2000 characters as a theme
    return extracted_text, document_theme
