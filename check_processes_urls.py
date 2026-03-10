import os
import re
import requests

directory = "knowledge_base/processes"
broken = []

for filename in os.listdir(directory):
    if filename.endswith(".md"):
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        urls = re.findall(r'<!-- url: (https?://[^\s>]+) -->', content)
        for url in urls:
            # remove anchor for checking
            check_url = url.split('#')[0]
            try:
                res = requests.get(check_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                if res.status_code >= 400:
                    broken.append((filename, url, res.status_code))
                else:
                    print(f"[OK] {filename}: {url}")
            except Exception as e:
                broken.append((filename, url, str(e)))

print("\n--- BROKEN URLs ---")
for b in broken:
    print(b)
if not broken:
    print("ALL URLs ARE WORKING 200 OK!")
