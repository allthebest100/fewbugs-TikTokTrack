import openai
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o"

# Placeholder image URL for visual grounding (can replace with real frame later)
PLACEHOLDER_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/9/99/Unofficial_JavaScript_logo_2.svg"

def answer(video_url, question, question_prompt, prompt_template):
    """
    Generates an answer using OpenAI GPT-4o (vision-capable).
    Uses a placeholder image for visual reference.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": prompt_template
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Video URL: {video_url}\n"
                                f"Question: {question}\n"
                                f"{question_prompt}".strip()
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": PLACEHOLDER_IMAGE_URL
                        }
                    }
                ]
            }
        ]

        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.3
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"ERROR: {type(e).__name__} - {str(e)}"
