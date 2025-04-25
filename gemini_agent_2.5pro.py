from google import genai
from google.genai import types
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

MAIN_GOOGLE_API_KEY = os.getenv("MAIN_GOOGLE_API_KEY")
ALT_GOOGLE_API_KEY = os.getenv("ALT_GOOGLE_API_KEY")
SPARE1_GOOGLE_API_KEY = os.getenv("SPARE1_GOOGLE_API_KEY")
SPARE2_GOOGLE_API_KEY = os.getenv("SPARE2_GOOGLE_API_KEY")
SPARE3_GOOGLE_API_KEY = os.getenv("SPARE3_GOOGLE_API_KEY")
SPARE4_GOOGLE_API_KEY = os.getenv("SPARE4_GOOGLE_API_KEY")
# SPARE5_GOOGLE_API_KEY = os.getenv("SPARE5_GOOGLE_API_KEY")
SPARE6_GOOGLE_API_KEY = os.getenv("SPARE6_GOOGLE_API_KEY")
SPARE7_GOOGLE_API_KEY = os.getenv("SPARE7_GOOGLE_API_KEY")
SPARE8_GOOGLE_API_KEY = os.getenv("SPARE8_GOOGLE_API_KEY")
SPARE9_GOOGLE_API_KEY = os.getenv("SPARE9_GOOGLE_API_KEY")
SPARE10_GOOGLE_API_KEY = os.getenv("SPARE10_GOOGLE_API_KEY")


# File paths
DATA_PATH = "data/all_data.csv"
VIDEO_DIR = "C:/Users/Ching Xi/Downloads/Benchmark-AllVideos-HQ-Encoded-challenge/Benchmark-AllVideos-HQ-Encoded-challenge"
OUTPUT_PATH = "agentic_output.csv"

# Prompt template (agentic)
SYSTEM_PROMPT = """
You are an expert AI system designed to analyze videos and answer user questions based only on what is visually shown.

Follow these steps carefully:

1. Observation: Describe the important actions, objects, and scenes in the video.
2. Decomposition: Break the user's question into one or more specific visual checks (e.g., number of objects, sequence of actions).
3. Verification: For each part, compare it to what was seen in the video and explain whether it is confirmed, contradicted, or unobservable.
4. Final Answer:
    - If the question asks for a description, provide a direct answer based on the video.
    - If it's a Yes/No question, clearly answer “Yes” or “No”, followed by a brief justification.
    - If it’s a counting question, provide the number and explain how it was counted.
    - If the video does not show enough, say: “❓ Not visually confirmable” and explain why.

⚠️ Do NOT assume anything not clearly visible. Be precise and explain your answer using only the video content.

"""

def generate_agentic_response(video_path, question, question_prompt):
    full_response = ""
    client = genai.Client(api_key=SPARE3_GOOGLE_API_KEY)

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(text=question + question_prompt),
                types.Part(
                    inline_data=types.Blob(
                        data=open(video_path, "rb").read(),
                        mime_type="video/*"
                    )
                )
            ],
        ),
    ]

    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="text/plain",
        system_instruction=SYSTEM_PROMPT,
    )


    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-pro-exp-03-25", contents=contents, config=config
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text
        return full_response
    except Exception as e:
        if "429" in str(e):
            return "RATE_LIMIT"
        print(f"ERROR: {type(e).__name__} - {str(e)}")


def main():
    df = pd.read_csv(DATA_PATH)

    # === Load existing results (if any) ===
    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        processed_qids = set(existing_df["qid"].astype(str))
        results = existing_df.to_dict("records")
    else:
        processed_qids = set()
        results = []

    for i, row in df.iterrows():
        qid = str(row["qid"])
        if qid in processed_qids:
            print(f"⏩ Skipping QID {qid} (already processed)")
            continue

        video_id = row["video_id"]
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"⚠️ Missing: {video_path}")
            continue

        print(f"\n🔍 Processing QID {qid}...")

        response = generate_agentic_response(
            video_path, row["question"], row.get("question_prompt", "")
        )

        if response == "RATE_LIMIT":
            break

        results.append({
            "qid": qid,
            "pred": response.strip()
        })

        # Save after every prediction
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)



if __name__ == "__main__":
    main()
