import PyPDF2 

import re

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

def process_pdf(uploaded_pdf):
    extracted_text = ""
    with open(uploaded_pdf.name, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                extracted_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text += page.extract_text()
    extracted_text = clean_extracted_text_v2(extracted_text)
            
    with open("extracted_text.txt", 'w+') as txt_file:
           txt_file.writelines(extracted_text)
            
    return extracted_text