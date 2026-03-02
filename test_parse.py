import json

BUTTONS_MARKER = "---BUTTONS---"

def parse_suggested_buttons(text: str) -> tuple:
    if BUTTONS_MARKER not in text:
        return text, []

    parts = text.rsplit(BUTTONS_MARKER, 1)
    clean_text = parts[0].rstrip()
    buttons_raw = parts[1].strip()
    
    # Strip markdown code blocks if the LLM added them
    if buttons_raw.startswith("```"):
        # Split by lines and remove first and last line if they are ```
        lines = buttons_raw.split('\n')
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            buttons_raw = '\n'.join(lines).strip()

    try:
        questions = json.loads(buttons_raw)
        if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
            return clean_text, questions
    except (json.JSONDecodeError, TypeError) as e:
        print(f"JSON Decode error: {e}")
        pass

    return clean_text, []

test_str_1 = """Здесь ответ бота.
---BUTTONS---
["Вопрос 1", "Вопрос 2"]
"""

test_str_2 = """Здесь ответ бота.
---BUTTONS---
```json
["Вопрос 1", "Вопрос 2"]
```
"""

print(parse_suggested_buttons(test_str_1))
print(parse_suggested_buttons(test_str_2))
