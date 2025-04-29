# app.py
import streamlit as st
import torch
from PIL import Image

from models.lite.model_loader import ModelLoader
from utils.summarizer import summarize_text
from utils.extractor import extract_medical_info
from utils.research import fetch_research
from utils.ocr_api import extract_text_ocr_space

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("MedAgent Lite - AI Assistant for Healthcare")

# Supported Models
model_names = {
    "BioGPT": "microsoft/BioGPT",
    "PubMedBERT": "dmis-lab/biobert-base-cased-v1.1",
    "Clinical-T5": "t5-small",
    "BioMed-RoBERTa": "allenai/biomed_roberta_base",
    "BlueBERT": "emilyalsentzer/Bio_ClinicalBERT"
}
model_loader = ModelLoader(model_names)

# Upload section
st.header("Upload a medical file:")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "png", "jpg", "jpeg"])
file_type = st.selectbox("Select file type", ["Text File (.txt)", "PDF", "Image"])

# Function to extract text
def extract_text(uploaded_file, file_type):
    if not uploaded_file:
        return None

    try:
        if file_type == "Text File (.txt)":
            return uploaded_file.read().decode("utf-8")

        elif file_type == "PDF":
            content, error = extract_text_ocr_space(uploaded_file, is_pdf=True)
            if error:
                st.error(f"OCR Error: {error}")
            return content

        elif file_type == "Image":
            content, error = extract_text_ocr_space(uploaded_file, is_pdf=False)
            if error:
                st.error(f"OCR Error: {error}")
            return content

    except Exception as e:
        st.error(f"Failed to read file: {str(e)}")
        return None

# Main execution
input_text = extract_text(uploaded_file, file_type)

if st.button("Process"):
    if input_text:
        with st.spinner("Running AI models..."):
            try:
                model, tokenizer = model_loader.load_model("Clinical-T5", device)
                summary = summarize_text(model, tokenizer, input_text, device)

                model, tokenizer = model_loader.load_model("BioMed-RoBERTa", device)
                medical_info = extract_medical_info(model, tokenizer, input_text, device)

                model, tokenizer = model_loader.load_model("BlueBERT", device)
                diagnosis_codes = extract_medical_info(model, tokenizer, input_text, device)

                model, tokenizer = model_loader.load_model("BioGPT", device)
                medical_note = summarize_text(model, tokenizer, input_text, device)

                research_papers = fetch_research(diagnosis_codes)

                st.subheader("AI Output")
                st.markdown(f"**Summary:** {summary}")
                st.markdown(f"**Medical Information:** {medical_info}")
                st.markdown(f"**Diagnosis Codes:** {diagnosis_codes}")
                st.markdown(f"**Medical Note:** {medical_note}")
                st.markdown("**Related Research Papers:**")
                st.json(research_papers)

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
    else:
        st.warning("No readable content found. Please upload a valid file.")
