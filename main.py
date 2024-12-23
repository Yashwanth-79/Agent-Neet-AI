import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from crewai import LLM, Agent, Task, Crew, Process
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# Load environment variables
load_dotenv()
import google.generativeai as genai
genai.configure(api_key="AIzaSyBGarpkJMffycGkdhU2ivf6YjgNGRYg138")

gemini_llm = genai.GenerativeModel(
  model_name="gemini-1.5-pro"
)

# Create CrewAI's LLM object
llm = LLM(
  model="gemini/gemini-1.5-pro", # or "gemini-1.5-pro" if you are using it, use model name instead of model_name
  api_key="AIzaSyBGarpkJMffycGkdhU2ivf6YjgNGRYg138", # or whatever your api key is
  provider="googleai"
)
# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None

def add_documents_to_vectorstore(pdf_path):
    global vector_store
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    if vector_store is None:
        vector_store = FAISS.from_documents(texts, embeddings)
    else:
        vector_store.add_documents(texts)

def search_vectorstore(query, k=3):
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)

def process_input(input_data, input_type):
    """Process either text or image input."""
    if input_type == "image":
        try:
            if isinstance(input_data, Image.Image):  # Check if it's a PIL Image object
                response = gemini_llm.generate_content([input_data])  # Use the correct genai object
            else:
                response = gemini_llm.generate_content([input_data])  # For raw image data
            return response.text
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return ""
    else: 
        response = gemini_llm.generate_content([input_data])  # Use the correct genai object
        return response.text


def create_subject_expert(subject):
    """Create a subject-specific NEET expert agent."""
    return Agent(
        role=f"NEET {subject} Expert",
        goal=f"Generate detailed {subject} questions following NEET examination standards",
        backstory=f"""
            You are an experienced NEET {subject} expert with deep understanding 
            of the examination pattern and curriculum. You excel at creating 
            questions that test conceptual understanding.
        """,
        allow_delegation=False,
        verbose=True,llm=llm
    )

def create_question_task(agent, subject, topic, processed_input, context):
    """Create a task for question generation with ReAct prompting."""
    context_text = "\n".join(context) if context else "No additional context"

    prompt = f"""Follow this ReAct (Reasoning and Acting) framework to generate a NEET question:

 Thought: Analyze the input and context
   - What is the key concept from the {subject} input?
   - What relevant information is available in the context?
   - What difficulty level is appropriate for NEET?

 Action: Plan the question structure
   - Identify the core concept to test
   - Determine question format (MCQ/numerical/theoretical)
   - Plan necessary calculations or reasoning steps

 Observation: Review available materials
   - Input material: {processed_input}
   - Context material: {context_text}
   - Topic focus: {topic}

 Question Generation: Generate 5 Question including the pdf and other commonly asked from your knowledge base
   - Create a clear, unambiguous question
   - Include any necessary diagrams or data
   - Provide 4 options if MCQ
   
 Answer Explanation:
   - Provide the correct answer
   - Give a detailed step-by-step solution
   - Explain the underlying concept
   - Add relevant NEET exam tips

Generate a complete NEET question following this framework.
"""

    return Task(
        description=prompt,
        expected_output="""A complete NEET question with:
                           1. Question text
                           2. Multiple choice options or solution approach
                           3. Correct answer
                           4. Detailed explanation
                           5. Conceptual insights""",
        agent=agent,llm=llm
    )

def generate_question(subject, topic, input_data, input_type, context=None):
    """Generate a NEET question based on input and context."""
    processed_input = process_input(input_data, input_type)
    expert = create_subject_expert(subject)
    task = create_question_task(expert, subject, topic, processed_input, context or [])

    crew = Crew(
        agents=[expert],
        tasks=[task],llm=llm
    )

    result = crew.kickoff()
    return result

# Streamlit App
st.title("NEET Mentor - AI Study Companion")
st.subheader("Upload Materials & Generate Questions")

# Study material upload
pdf_file = st.file_uploader("Upload your study materials (PDF)", type="pdf")
if pdf_file:
    with st.spinner("Processing your study materials..."):
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())
        add_documents_to_vectorstore("temp.pdf")
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
                search_results = search_vectorstore(f"NEET {subject} {topic}")
                context = [doc.page_content for doc in search_results]

            result = generate_question(subject, topic, input_data, input_type.lower(), context)

            st.markdown("### Generated Question")
            st.write(result)
    else:
        st.warning("Please provide either text input or upload an image.")
