import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("MAIN_GOOGLE_API_KEY")

MODEL_NAME = "gemini-2.5-pro-preview-03-25"

# Configure Gemini API
genai.configure(api_key=API_KEY)

def answer(video_url, question, question_prompt, prompt_template):
    """
    Generates an answer using Gemini 2.5 Pro based on question, video URL, and system prompt.
    """
    try:
        model = genai.GenerativeModel(MODEL_NAME)

        user_input = (
            f"{prompt_template}\n\n"
            f"Video URL: {video_url}\n"
            f"Question: {question}\n"
            f"{question_prompt}".strip()
        )

        response = model.generate_content(
            [user_input],
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 512,
                "top_p": 1,
                "top_k": 1
            }
        )

        return response.text.strip()

    except Exception as e:
        return f"ERROR: {type(e).__name__} - {str(e)}"
