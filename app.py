import os
import time
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def extract_text_from_file(file_path: str) -> str:
    """Reads and extracts text from both TXT and PDF files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dorakaledu: {file_path}")

    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        extracted_text = ""
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            extracted_text += f"\n--- Page {page_num} ---\n{text}"
        return extracted_text.strip()
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

file_to_read = "sample.txt"
print(f"Reading content from {file_to_read}...")
doc_context = extract_text_from_file(file_to_read)

question = "Summarize the project highlights in 2 bullet points."
prompt = f"""
Answer the following question based solely on the provided document context.

--- Context ---
{doc_context}

--- Question ---
{question}
"""

print("Sending context to Gemini...")

# Retry mechanism for 503 high demand errors
max_retries = 3
for attempt in range(max_retries):
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        print("\n--- Answer ---")
        print(response.text)
        break
    except Exception as e:
        if "503" in str(e) and attempt < max_retries - 1:
            print(f"Server busy ga undi. 3 seconds lo malli try chestunna (Attempt {attempt + 2}/{max_retries})...")
            time.sleep(3)
        else:
            raise e