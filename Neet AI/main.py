import os
import streamlit as st
from PIL import Image
from utils.document_processor import DocumentProcessor
from agents.expert_agent import generate_question

# Initialize DocumentProcessor
doc_processor = DocumentProcessor()

# Streamlit App
st.title("NEET Mentor - AI Study Companion")
st.subheader("Upload Materials & Generate Questions")

# Study material upload
pdf_file = st.file_uploader("Upload your study materials (PDF)", type="pdf")
if pdf_file:
    with st.spinner("Processing your study materials..."):
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())
        doc_processor.add_documents_to_vectorstore("temp.pdf")
        os.remove("temp.pdf")
        st.success("Materials processed successfully!")

# Input selection
input_type = st.radio("Select input type:", ["Text", "Image"])
input_data = None
if input_type == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        input_data = Image.open(img_file)
        st.image(input_data, caption="Uploaded Image")
        st.info("The image will be analyzed to generate relevant questions.")
else:
    input_data = st.text_area("Enter your text input:", placeholder="Enter concept or topic description...")

col1, col2 = st.columns(2)
with col1:
    subject = st.selectbox("Select Subject", ["Physics", "Chemistry", "Biology"])
with col2:
    topic = st.text_input("Enter specific topic (optional)", placeholder="e.g., Thermodynamics")

if st.button("Generate Question"):
    if input_data:
        with st.spinner("Generating NEET question..."):
            context = []
            if topic:
                search_results = doc_processor.search_vectorstore(f"NEET {subject} {topic}")
                context = [doc.page_content for doc in search_results]

            result = generate_question(subject, topic, input_data, input_type.lower(), context)

            st.markdown("### Generated Question")
            st.write(result)
    else:
        st.warning("Please provide either text input or upload an image.")