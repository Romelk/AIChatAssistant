from PyPDF2 import PdfReader

def split_text(text, chunk_size=1000):
    """Splits text into chunks of defined size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def extract_pdf_text(file_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

if __name__ == "__main__":
    document_text = extract_pdf_text("your_document.pdf")
    text_chunks = split_text(document_text)
    print("Text split into chunks: ", text_chunks)

