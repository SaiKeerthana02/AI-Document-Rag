import os
from dotenv import load_dotenv
from google import genai

# .env file nunchi secret key ni load chestundi
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Connecting to Gemini using .env...")

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Explain what an AI Document Assistant does in 2 simple sentences.",
)

print("\n--- Output ---")
print(response.text)