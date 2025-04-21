import os
import argparse
import pandas as pd
from difflib import SequenceMatcher

# === Utility Functions ===

def normalize(text):
    if not isinstance(text, str):
        return ""
    return text.strip().lower().replace("answer:", "").strip()

def string_match(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() >= threshold

# === Metric Calculators ===

def evaluate_accuracy(df):
    total = len(df)
    correct = 0
    for _, row in df.iterrows():
        pred = normalize(row.get("pred", ""))
        answer = normalize(row.get("answer", ""))
        if answer and string_match(pred, answer):
            correct += 1
    return round((correct / total) * 100, 2) if total > 0 else 0

def evaluate_misleading_resistance(df):
    total = len(df)
    safe_phrases = ["cannot", "unclear", "not shown", "not visible", "invalid"]
    resistant = 0
    for _, row in df.iterrows():
        pred = normalize(row.get("pred", ""))
        if any(phrase in pred for phrase in safe_phrases):
            resistant += 1
    return round((resistant / total) * 100, 2) if total > 0 else 0

# === Main Evaluation Dispatcher ===

def evaluate(qtype, model_name):
    pred_path = f"outputs/{qtype}_results_{model_name}.csv"

    if qtype == "primary":
        truth_path = "reina/question datasets/Primary Open-ended Question.csv"
    elif qtype == "paraphrased":
        truth_path = "reina/question datasets/Paraphrased Open-ended Question.csv"
    elif qtype == "wrongly_led":
        truth_path = "reina/question datasets/Wrongly-led Open-ended Question.csv"
    elif qtype == "correctly_led":
        truth_path = "reina/question datasets/Correctly-led Open-ended Question.csv"
    elif qtype == "mcq":
        truth_path = "reina/question datasets/mcq_cleaned.csv"
    else:
        raise ValueError("❌ Invalid question type.")

    # Check file existence
    if not os.path.exists(pred_path) or not os.path.exists(truth_path):
        raise FileNotFoundError("❌ Required input files not found.")

    pred_df = pd.read_csv(pred_path)
    truth_df = pd.read_csv(truth_path)

    df = pd.merge(truth_df, pred_df, on="qid", how="inner")

    print(f"\n📊 Evaluation Summary — {model_name.upper()} | {qtype.upper()}")
    print("-" * 45)

    if qtype in ["primary", "correctly_led", "mcq"]:
        acc = evaluate_accuracy(df)
        print(f"✅ Accuracy: {acc}%")
    elif qtype == "wrongly_led":
        score = evaluate_misleading_resistance(df)
        print(f"🛡️ Misleading Resistance: {score}%")
    elif qtype == "paraphrased":
        print("🔄 Paraphrased consistency: (TBD) ✏️ Not yet implemented.")
    else:
        print("❌ Unsupported evaluation type.")

    print("-" * 45)

# === CLI Interface ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--model", required=True, choices=["gpt4o", "gemini"], help="Model used for predictions")
    parser.add_argument("--type", required=True, choices=[
        "primary", "paraphrased", "wrongly_led", "correctly_led", "mcq"
    ], help="Question type to evaluate")

    args = parser.parse_args()
    evaluate(args.type, args.model)
