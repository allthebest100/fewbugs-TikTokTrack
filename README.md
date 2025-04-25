📽️ Project Description: Agentic Video Question Answering using Gemini 2.5 Pro
This project implements a video-based question-answering pipeline using Google's Gemini 2.5 Pro multimodal API. It focuses on an agentic reasoning prompt design, enabling the model to carefully observe video content, decompose the user's question, verify visual evidence, and generate grounded, step-by-step answers. The model is strictly constrained to respond only based on visual information present in the video, enhancing reliability for downstream evaluation tasks.

- The pipeline processes hundreds of .mp4 benchmark videos alongside a corresponding CSV of question-answer pairs. For each video-question pair, the system:
- Loads the local video file.
- Sends it to Gemini 2.5 Pro along with the user question and an agentic system prompt.
- Streams back the response in real-time.
- Saves the results to a checkpointed output file to prevent reprocessing.

💡 Problem Statement
The task is to evaluate the reasoning ability of a multimodal LLM over visual input by answering diverse questions about pre-recorded videos. This includes yes/no questions, counting tasks, temporal reasoning, and factual queries. The challenge is to ensure strict visual-grounding, where the model must not hallucinate or infer unseen content. The goal is to achieve higher accuracy by improving prompt structure and input quality in a scalable pipeline.

🛠️ Development Tools
- Language: Python 3.10+
- Environment Manager: dotenv for API key management
- IDE: Visual Studio Code

🔗 APIs Used
- Google Generative AI (Gemini) via the google.genai SDK
- Model: gemini-2.5-pro-exp-03-25
- API endpoint supports video input and streaming response generation

🧠 Assets Used
- Video Dataset: Local .mp4 videos (Benchmark-AllVideos-HQ-Encoded-challenge)
- CSV Metadata: all_data.csv — contains qid, video_id, question, and optionally question_prompt
- Prompt: Custom-designed agentic reasoning system prompt, optimized for decomposed logical steps

📦 Libraries and Packages
- google-generativeai
- pandas
- python-dotenv
- os, time (built-in)

Install dependencies:
- pip install -r requirements.txt

🔁 Functionality Overview
- Skips previously processed questions (agentic_output.csv used as checkpoint)
- Handles local video loading and MIME encoding
- Implements automatic rate-limit detection and pause handling (429 Too Many Requests)
- Saves after every single prediction to prevent data loss
- Designed to support rotation between multiple API keys

🔗 Public GitHub Repository
👉 [GitHub Repository Link](https://github.com/allthebest100/fewbugs-TikTokTrack)