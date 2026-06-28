import os
import traceback

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

try:
    r = model.generate_content("Xin chào")
    print("OK", type(r))
    print("has text", hasattr(r, "text"))
    print("text=", repr(getattr(r, "text", None)))
except Exception as e:
    print("EXC", type(e).__name__)
traceback.print_exc()
