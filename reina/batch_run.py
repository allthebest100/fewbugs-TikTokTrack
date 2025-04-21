import subprocess

QUESTION_TYPES = ["primary", "paraphrased", "wrongly_led", "correctly_led", "mcq"]

def run_all(model_name):
    for qtype in QUESTION_TYPES:
        print(f"\n🔄 Running inference: {model_name.upper()} | {qtype}")
        subprocess.run([
            "python", "reina/run_inference.py",
            "--model", model_name,
            "--type", qtype
        ])

        print(f"✅ Inference complete: {qtype} | Now evaluating...\n")

        subprocess.run([
            "python", "reina/evaluate.py",
            "--model", model_name,
            "--type", qtype
        ])

    print("\n🎉 All tasks complete!")

if __name__ == "__main__":
    print("🎯 Batch Mode: Run all 5 question types")
    model = input("👉 Choose model [gpt4o / gemini]: ").strip().lower()

    if model not in ["gpt4o", "gemini"]:
        print("❌ Invalid model name. Please choose either 'gpt4o' or 'gemini'.")
    else:
        run_all(model)
