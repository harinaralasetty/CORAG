import PyPDF2 

def process_pdf(uploaded_pdf):
    extracted_text = ""
    with open(uploaded_pdf.name, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                extracted_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text += page.extract_text()
                print(f"Extrracted text: {extracted_text}")
            
    return extracted_text