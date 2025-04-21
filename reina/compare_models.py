import pandas as pd
import subprocess
import os

QUESTION_TYPES = ["primary", "paraphrased", "wrongly_led", "correctly_led", "mcq"]
MODELS = ["gpt4o", "gemini"]

# Reuse evaluation logic via subprocess and capture output
def get_score(qtype, model):
    try:
        result = subprocess.run(
            ["python", "reina/evaluate.py", "--model", model, "--type", qtype],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout

        if "Accuracy" in output:
            score = float(output.split("Accuracy:")[1].split("%")[0].strip())
        elif "Misleading Resistance" in output:
            score = float(output.split("Resistance:")[1].split("%")[0].strip())
        else:
            score = None  # For paraphrased (TBD)
        return score
    except Exception as e:
        return f"ERR: {e.__class__.__name__}"

def compare_models():
    comparison = []

    for qtype in QUESTION_TYPES:
        row = {"question_type": qtype}
        for model in MODELS:
            score = get_score(qtype, model)
            row[model] = score
        comparison.append(row)

    df = pd.DataFrame(comparison)
    print("\n📊 Model Comparison Table (GPT-4o vs Gemini)\n")
    print(df.to_markdown(index=False))

    # Optional: Save to file
    os.makedirs("evaluation_reports", exist_ok=True)
    df.to_csv("evaluation_reports/model_comparison.csv", index=False)
    print("\n✅ Saved to: evaluation_reports/model_comparison.csv")

if __name__ == "__main__":
    compare_models()
