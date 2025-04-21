import pandas as pd
import os

# === CONFIG ===
QUESTION_TYPES = ["primary", "paraphrased", "wrongly_led", "correctly_led", "mcq"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")  # absolute path to reina/outputs

def generate_submission(preferred_model="gemini"):
    dfs = []
    submission_name = f"reina-submission-{preferred_model}.csv"
    submission_path = os.path.join(os.path.dirname(__file__), submission_name)

    print(f"\n📝 Generating submission file for model: {preferred_model}")
    for qtype in QUESTION_TYPES:
        filename = f"{qtype}_results_{preferred_model}.csv"
        result_file = os.path.join(OUTPUT_DIR, filename)
        print(f"🔍 Looking for: {result_file}")

        if not os.path.exists(result_file):
            print(f"⚠️ Skipped {qtype}: result file not found for {preferred_model}")
            continue

        df = pd.read_csv(result_file)
        if "qid" not in df.columns or "pred" not in df.columns:
            print(f"❌ Invalid format in {filename}")
            continue

        dfs.append(df[["qid", "pred"]])

    if not dfs:
        print("❌ No valid results found. Submission not generated.")
        return

    final_df = pd.concat(dfs).sort_values("qid")
    final_df.to_csv(submission_path, index=False)
    print(f"✅ Submission saved to: {submission_path}")

if __name__ == "__main__":
    generate_submission(preferred_model="gemini")
    generate_submission(preferred_model="gpt4o")  # Optional second version
