# utils/file_handler.py
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import io

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")
