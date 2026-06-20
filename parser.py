"""
parser.py
Handles extracting raw text from uploaded resume files (PDF / DOCX).
"""

import pdfplumber
from docx import Document
import os


def extract_text(filepath):
    """
    Detects file type by extension and extracts plain text.
    Returns: (text, error_message)
    """
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".pdf":
            return extract_text_pdf(filepath), None
        elif ext == ".docx":
            return extract_text_docx(filepath), None
        else:
            return None, "Unsupported file type. Please upload a PDF or DOCX file."
    except Exception as e:
        return None, f"Could not read file: {str(e)}"


def extract_text_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        # This usually means the PDF is a scanned image, not real text
        raise ValueError(
            "No selectable text found. This PDF might be a scanned image rather than text-based."
        )
    return text


def extract_text_docx(filepath):
    doc = Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])

    if not text.strip():
        raise ValueError("The DOCX file appears to be empty.")
    return text
