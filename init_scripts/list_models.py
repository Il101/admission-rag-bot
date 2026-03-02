from google import genai
import os

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found")
        return
    
    client = genai.Client(api_key=api_key)
    print("Listing available models via google-genai...")
    try:
        # The new SDK has list_models but the structure is different
        # Let's try to list them and find embedding capability
        for m in client.models.list():
            # Check for embedding support
            # The model object has supported_methods in the old one, let's see for the new one
            print(f"Model: {m.name}, Display Name: {m.display_name}")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    main()
