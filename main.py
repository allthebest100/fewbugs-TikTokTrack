from google import genai
from google.genai import types
from dotenv import load_dotenv
import pandas as pd
import os
import time

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Check your .env file.")

# API limits
REQUESTS_PER_MIN = 15  # 15 requests per minute
SLEEP_TIME = 60 / REQUESTS_PER_MIN  # Time per request

def generate(video_url, question, question_prompt):
    """Generates a response for a given video URL and question using Gemini API, respecting rate limits."""
    
    full_response = ""
    client = genai.Client(api_key=GOOGLE_API_KEY)

    model = "gemini-2.0-flash"  
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=video_url,
                    mime_type="video/*",
                ),
                types.Part.from_text(text=question + question_prompt),
            ],
        ),
    ]

    system_instruction = """
    You are an AI that accurately analyzes the content of a given video. Your task is to evaluate whether a user's question correctly describes the events in the video.
    
    1. Watch the entire video carefully and understand the sequence of events.
    2. Analyze the user's question and compare it with what actually happens in the video.
    3. Provide a clear answer addressing the user's question based *only* on the visual evidence in the video. Explain your reasoning by referencing specific moments if necessary.
    """
    
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="text/plain",
        system_instruction=system_instruction,
    )

    try:
        for chunk in client.models.generate_content_stream(
            model=model, contents=contents, config=generate_content_config
        ):
            if chunk.text:
                print(chunk.text, end="")
                full_response += chunk.text
        return full_response
    except Exception as e:
        return f"ERROR: {type(e).__name__} - {str(e)}"

def import_data():
    """Imports video data from a CSV file."""
    try:
        data = pd.read_csv("data/valid_links_only.csv")
        return data
    except FileNotFoundError:
        raise FileNotFoundError("The file 'data/valid_links_only.csv' was not found.")

if __name__ == "__main__":
    data = import_data()
    total_items = len(data)

    print(f"--- Starting processing for {total_items} videos ---")

    results = []
    request_count = 0

    for i, row in data.iterrows():
        start_time = time.time()
        qid = row["qid"]
        video_url = row["youtube_url"]
        question = row["question"]
        question_prompt = row["question_prompt"]

        print(f"\nProcessing item {i + 1}/{total_items}: QID {qid}")

        response = generate(video_url, question, question_prompt)
        results.append((qid, response))

        elapsed_time = time.time() - start_time
        print(f"\n-> Done with QID {qid} in {elapsed_time:.2f} seconds.")

        # Increment request counter
        request_count += 1

        # Respect rate limit: pause every 15 requests
        if request_count % REQUESTS_PER_MIN == 0:
            print(f"Rate limit reached. Sleeping for 60 seconds...")
            time.sleep(60)

        # Short pause between requests
        time.sleep(SLEEP_TIME)

    # Save results
    df = pd.DataFrame(results, columns=["qid", "pred"])
    df.to_csv("cx-submission.csv", index=False)
    print("Results saved to cx-submission.csv.")
