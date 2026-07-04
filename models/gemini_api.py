import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_guidance(problem, emotion):

    prompt = f"""
You are an empathetic AI Learning Assistant.

Student Emotion:
{emotion}

Student Problem:
{problem}

Respond in this format:

1. Emotion Analysis
2. Simple Explanation
3. Encouragement
4. Next Steps
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text