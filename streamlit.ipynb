import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from PyPDF2 import PdfReader

file_path = "/content/drive/MyDrive/Resume.pdf"

# Load the BART model and tokenizer for summarization
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + "\n"
    return text.strip()

# Function to summarize text using BART
def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    st.title("PDF Summarizer with BART")

    # Upload PDF via Streamlit
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        text_from_pdf = extract_text_from_pdf(uploaded_file)
        st.subheader("Original Text:")
        st.write(text_from_pdf[:2000])  # Display first 2000 characters for readability

        # Summarize the extracted text
        summary = summarize_text(text_from_pdf)
        st.subheader("Summarized Text:")
        st.write(summary)

if __name__ == "__main__":
    main()
