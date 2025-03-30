from google import genai
from google.genai import types
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# def generate():
#     client = genai.Client(
#         api_key=os.environ.get("GEMINI_API_KEY"),
#     )

#     model = "gemini-2.5-pro-exp-03-25"
#     contents = [
#         types.Content(
#             role="user",
#             parts=[
#                 types.Part.from_uri(
#                     file_uri="https://youtu.be/FdkXmy42Qv8",
#                     mime_type="video/*",
#                 ),
#                 types.Part.from_text(text="""In the video, does the man first teleport to the bookshelf, then to the cardboard box, then to the couch, and finally return to under the blanket?"""),
#             ],
#         ),
#     ]
#     generate_content_config = types.GenerateContentConfig(
#         temperature=0,
#         response_mime_type="text/plain",
#     )

#     for chunk in client.models.generate_content_stream(
#         model=model,
#         contents=contents,
#         config=generate_content_config,
#     ):
#         print(chunk.text, end="")

# if __name__ == "__main__":
#     generate()

data = pd.read_parquet("data/test-00000-of-00001.parquet")
print(data.head(10))