import os

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)

for i, m in enumerate(genai.list_models()):
    print(i, getattr(m, "name", repr(m)))
    if i >= 30:
        break
