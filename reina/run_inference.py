import os
import argparse
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env variables
load_dotenv()
GEMINI_API_KEY = os.getenv("MAIN_GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Mapping question types to CSV paths
QUESTION_TYPES = {
    "primary": "reina/question datasets/Primary Open-ended Question.csv",
    "paraphrased": "reina/question datasets/Paraphrased Open-ended Question.csv",
    "wrongly_led": "reina/question datasets/Wrongly-led Open-ended Question.csv",
    "correctly_led": "reina/question datasets/Correctly-led Open-ended Question.csv",
    "mcq": "reina/question datasets/mcq_cleaned.csv"
}

PROMPT_DIR = "reina/prompt"
OUTPUT_DIR = "outputs"

# Load prompt text from prompt files
def load_prompt(qtype):
    path = os.path.join(PROMPT_DIR, f"{qtype}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Load the model function depending on choice
def load_model(model_choice):
    if model_choice == "gemini":
        from models import gemini_api
        return gemini_api.answer
    elif model_choice == "gpt4o":
        from models import gpt4v_api
        return gpt4v_api.answer
    else:
        raise ValueError("❌ Unsupported model. Use 'gpt4o' or 'gemini'.")

# Run inference on a dataset
def run_inference(model_choice, qtype):
    input_file = QUESTION_TYPES[qtype]
    output_file = os.path.join(OUTPUT_DIR, f"{qtype}_results_{model_choice}.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n🚀 Inference Start: {model_choice.upper()} on '{qtype}'")
    df = pd.read_csv(input_file)
    prompt = load_prompt(qtype)
    model_fn = load_model(model_choice)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        qid = row.get("qid")
        question = row.get("question")
        video_url = row.get("youtube_url")
        question_prompt = row.get("question_prompt", "")

        try:
            response = model_fn(video_url, question, question_prompt, prompt)
        except Exception as e:
            response = f"ERROR: {type(e).__name__} - {str(e)}"

        results.append({
            "qid": qid,
            "pred": response
        })

        time.sleep(1)  # Optional: avoid rate limits

    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n✅ Inference complete. Results saved to: {output_file}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on question types using GPT-4o or Gemini.")
    parser.add_argument("--model", required=True, choices=["gpt4o", "gemini"], help="Model: gpt4o or gemini")
    parser.add_argument("--type", required=True, choices=list(QUESTION_TYPES.keys()), help="Question type to run")
    args = parser.parse_args()

    run_inference(model_choice=args.model, qtype=args.type)
