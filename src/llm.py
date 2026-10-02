
import os
import time
from google.genai import errors
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

# Read the API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# Model to use
MODEL_NAME = "gemini-3.8-flash"




def generate_answer(question, context):
    """Generate an answer, retrying temporary API failures."""

    prompt = f"""
You are TechCare's customer support assistant.

Rules:
1. Answer using only the supplied context.
2. Do not invent company policies or facts.
3. If the context does not contain the answer, say:
   "I couldn't find that information in the TechCare knowledge base."
4. Keep the answer clear and concise.

Context:
{context}

Customer question:
{question}
"""

    for attempt in range(3):
        try:
            response = client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
                store=False
            )

            if response.output_text:
                return response.output_text.strip()

            return "Sorry, I couldn't generate an answer."

        except errors.APIError as error:
            status = getattr(error, "code", None)

            # Retry only temporary server/rate-limit errors
            if status not in (429, 500, 502, 503, 504):
                raise

            if attempt == 2:
                raise

            wait_seconds = 2 ** (attempt + 1)

            print(
                f"Gemini temporarily unavailable "
                f"(HTTP {status}). Retrying in "
                f"{wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

