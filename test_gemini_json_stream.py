import asyncio
import os
import json
from google import genai
from google.genai import types

async def test_json_stream():
    api_key = ""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "GOOGLE_API_KEY=" in line:
                    api_key = line.split("=")[1].strip()
                    break
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    api_key = "".join(c for c in api_key if ord(c) < 128)
    client = genai.Client(api_key=api_key)
    
    schema = {
        "type": "OBJECT",
        "properties": {
            "answer": {"type": "STRING"},
            "suggested_questions": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["answer", "suggested_questions"]
    }
    
    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant. Always respond in valid JSON format according to the schema.",
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=schema
    )
    
    print("Testing streaming JSON with gemini-2.0-flash...")
    try:
        stream = client.models.generate_content_stream(
            model='gemini-2.0-flash',
            contents="Tell me about A2 level German in 2 paragraphs.",
            config=config
        )
        accumulated = ""
        for chunk in stream:
            if chunk.text:
                accumulated += chunk.text
                print(f"CHUNK: {chunk.text!r}")
        print(f"\nFinal accumulated: {accumulated}")
        try:
            parsed = json.loads(accumulated)
            print("Successfully parsed final JSON!")
            print(f"Answer length: {len(parsed['answer'])}")
            print(f"Buttons: {parsed['suggested_questions']}")
        except:
            print("Failed to parse accumulated text as JSON.")
    except Exception as e:
        print(f"Streaming error: {e}")

if __name__ == "__main__":
    asyncio.run(test_json_stream())
