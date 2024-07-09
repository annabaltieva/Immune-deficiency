import faiss
from sentence_transformers import SentenceTransformer, util
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import fitz  # PyMuPDF
import os
import numpy as np
import sys

# Initialize Sentence Transformer model for retrieval
retriever_model = SentenceTransformer('all-mpnet-base-v2')

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text

def preprocess_text(text):
    # Remove extra spaces, newlines, etc.
    return ' '.join(text.split())

# Index all PDFs
pdf_dir = "data/"
pdf_paths = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

print(f"Found {len(pdf_paths)} PDF files to process.")

# Extract text and compute embeddings
documents = []
doc_embeddings = []
for i, pdf_path in enumerate(pdf_paths):
    print(f"Processing file {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    text = preprocess_text(text)
    if not text.strip():
        print(f"Warning: No text found in {pdf_path}")
        continue
    print(f"Extracted text from document {i+1}: {text[:200]}...")  # Show the first 200 characters of the extracted text
    documents.append((f"doc_{i+1}", text))
    embedding = retriever_model.encode(text)
    doc_embeddings.append(embedding)
    print(f"Encoded document {i+1} with embedding shape {embedding.shape}")

# Check if embeddings were generated
if not doc_embeddings:
    raise ValueError("No document embeddings were generated. Please check the PDF files and ensure they contain extractable text.")

doc_embeddings = np.array(doc_embeddings)
num_dimensions = doc_embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(num_dimensions)
index.add(doc_embeddings)

def retrieve_documents(query, k=3):
    query_embedding = retriever_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [(documents[idx][0], documents[idx][1], distances[0][i]) for i, idx in enumerate(indices[0])]
    return retrieved_docs

# Load your own trained DistilBERT model for classification
model_path = 'results/model'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Function to classify text using your model
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probabilities = outputs.logits.softmax(dim=-1)
    predicted_class = probabilities.argmax().item()
    return predicted_class, probabilities

def read_documents(query, documents):
    answers = []
    for doc_id, text, distance in documents:
        predicted_class, probabilities = classify_text(text)
        answer = "YES" if predicted_class == 1 else "NO"
        answers.append((doc_id, answer, probabilities, distance))
    return answers

def rag_pipeline(query):
    retrieved_docs = retrieve_documents(query)
    answers = read_documents(query, retrieved_docs)
    return answers

if __name__ == "__main__":
    static_question = "According to the text, classify with YES or NO the immune deficiency:"
    while True:
        print(static_question)
        therapy_text = input("Enter therapy text (or 'exit' to quit): ")
        if therapy_text.lower() == 'exit':
            print("Exiting program.")
            sys.exit()
        result = rag_pipeline(therapy_text)
        # Aggregate results
        yes_count = sum(1 for _, answer, _, _ in result if answer == "YES")
        no_count = sum(1 for _, answer, _, _ in result if answer == "NO")
        final_answer = "YES" if yes_count > no_count else "NO"
        print(f"Answer: {final_answer}")
        print("Processing complete. The program will now exit.")
        sys.exit()
