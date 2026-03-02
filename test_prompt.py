import asyncio
from bot.memory import build_journey_state_prompt  # Assuming this or similar exists
from bot.utils import get_config
from langchain_core.prompts import load_prompt

# Instead of running the full DB, let's just inspect the prompt we load
import yaml
with open('configs/prompts/gemini-2.5-flash.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Prompt loaded successfully. Check if the rules are present:")
for msg in config['rag_prompt']['messages']:
    if 'content' in msg:
        print("--- CONTENT ---")
        print(msg['content'])

