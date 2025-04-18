from google import genai
from google.genai import types
from dotenv import load_dotenv
import pandas as pd
import os
import time

load_dotenv()

MAIN_GOOGLE_API_KEY = os.getenv("MAIN_GOOGLE_API_KEY")
ALT_GOOGLE_API_KEY = os.getenv("ALT_GOOGLE_API_KEY")
SPARE1_GOOGLE_API_KEY = os.getenv("SPARE1_GOOGLE_API_KEY")
SPARE2_GOOGLE_API_KEY = os.getenv("SPARE2_GOOGLE_API_KEY")
SPARE3_GOOGLE_API_KEY = os.getenv("SPARE3_GOOGLE_API_KEY")
SPARE4_GOOGLE_API_KEY = os.getenv("SPARE4_GOOGLE_API_KEY")
SPARE5_GOOGLE_API_KEY = os.getenv("SPARE5_GOOGLE_API_KEY")

# API limits
REQUESTS_PER_MIN = 5
SLEEP_TIME = 60 / REQUESTS_PER_MIN

# Config
BATCH_SIZE = 5
DAILY_LIMIT = 25
SLEEP_AFTER_BATCH = 30
INPUT_FILE = "data/valid_links_only.csv"
OUTPUT_FILE = "cx-submission-gemini-2.5-pro.csv"

def generate(video_url, question, question_prompt):
    """Generates a response for a given video URL and question using Gemini API, respecting rate limits."""
    
    full_response = ""
    client = genai.Client(api_key=SPARE3_GOOGLE_API_KEY)

    model = "gemini-2.5-pro-exp-03-25"  
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(file_uri=video_url,mime_type="video/*"),
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
    try:
        return pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{INPUT_FILE}' was not found.")

# === Load Already Processed QIDs (Checkpoint) ===
def load_existing_results():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_csv(OUTPUT_FILE)
    else:
        return pd.DataFrame(columns=["qid", "pred"])

if __name__ == "__main__":
    data = import_data()
    existing = load_existing_results()
    processed_qids = set(existing["qid"].astype(str))

    total_items = len(data)
    daily_request_count = 0
    batch = []

    print(f"--- Starting processing up to {DAILY_LIMIT} videos ---")

    for i, row in data.iterrows():
        qid = str(row["qid"])
        if qid in processed_qids:
            print(f"⏩ Skipping QID {qid} (already processed)")
            continue

        if daily_request_count >= DAILY_LIMIT:
            print("⛔️ Reached daily limit of 25 requests. Stopping.")
            break

        print(f"\n🚀 Processing {i + 1}/{total_items} - QID {qid}")
        start_time = time.time()

        response = generate(
            row["youtube_url"], row["question"], row["question_prompt"]
        )

        batch.append((qid, response))
        daily_request_count += 1

        elapsed = time.time() - start_time
        print(f"\n✅ Done with QID {qid} in {elapsed:.2f}s.")

        # Save and sleep after every batch of 5
        if len(batch) >= BATCH_SIZE:
            print(f"\n💾 Saving batch of {BATCH_SIZE} results...")
            pd.DataFrame(batch, columns=["qid", "pred"]).to_csv(
                OUTPUT_FILE, mode="a", header=not os.path.exists(OUTPUT_FILE), index=False
            )
            batch = []
            print(f"⏳ Sleeping for {SLEEP_AFTER_BATCH} seconds to respect RPM limit...")
            time.sleep(SLEEP_AFTER_BATCH)

    # Save any remaining batch
    if batch:
        print(f"\n💾 Saving final batch of {len(batch)} results...")
        pd.DataFrame(batch, columns=["qid", "pred"]).to_csv(
            OUTPUT_FILE, mode="a", header=not os.path.exists(OUTPUT_FILE), index=False
        )

    print("🎉 All done for today.")
