import os
from dotenv import load_dotenv
import google.generativeai as genai
from crewai import LLM

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_llm = genai.GenerativeModel(
    model_name="gemini-1.5-pro"
)

# Create CrewAI's LLM object
crew_llm = LLM(
    model="gemini/gemini-1.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
    provider="googleai"
)