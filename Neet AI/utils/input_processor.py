from PIL import Image
import streamlit as st
from config.llm_config import gemini_llm

def process_input(input_data, input_type):
    """Process either text or image input."""
    try:
        if input_type == "image":
            if isinstance(input_data, Image.Image):
                response = gemini_llm.generate_content([input_data])
            else:
                response = gemini_llm.generate_content([input_data])
            return response.text
        else:
            response = gemini_llm.generate_content([input_data])
            return response.text
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")
        return ""