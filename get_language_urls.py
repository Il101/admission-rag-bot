import re
import os
import json

directory = "/Users/iliazarikov/Documents/Python Projects/Агент по поступлению/knowledge_base/language"
all_urls = []

for filename in os.listdir(directory):
    if filename.endswith(".md"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            # Find URLs (http or https)
            urls = re.findall(r'https?://[^\s)\]"\'<>]+', content)
            
            for url in urls:
                # Basic cleanup
                url = url.rstrip('.,;:!?')
                if url not in all_urls:
                    all_urls.append(url)

print(json.dumps(all_urls, indent=2))
