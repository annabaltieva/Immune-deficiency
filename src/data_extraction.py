import fitz  # PyMuPDF
import re
import os
import json

def extract_therapy_section(pdf_path):
    doc = fitz.open(pdf_path)
    therapy_text = ""
    for page in doc:
        text = page.get_text("text")
        if "Терапия" in text:
            start_idx = text.index("Терапия")
            end_idx = text.find("\n\n", start_idx)
            if end_idx == -1:
                end_idx = len(text)
            therapy_text = text[start_idx:end_idx].strip()
            break
    return therapy_text

def extract_all_therapies(pdf_dir):
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files in directory: {pdf_dir}")
    therapies = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        print(f"Extracting text from: {pdf_path}")
        therapy_text = extract_therapy_section(pdf_path)
        if therapy_text:
            therapies.append(therapy_text)
    return therapies

if __name__ == "__main__":
    pdf_dir = "/Users/giordani/Модели/immune_deficiency_detection/data/"
    print("Current working directory:", os.getcwd())
    print(f"Starting extraction in directory: {pdf_dir}")
    therapies = extract_all_therapies(pdf_dir)
    output_file = '/Users/giordani/Модели/immune_deficiency_detection/data/extracted_therapies.json'
    with open(output_file, 'w') as f:
        json.dump(therapies, f, ensure_ascii=False, indent=4)
    print(f"Extracted therapies saved to {output_file}")
    for therapy in therapies:
        print(therapy)
