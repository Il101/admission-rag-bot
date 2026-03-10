import urllib.request
import urllib.parse
import re

queries = [
    "site:oead.at/en employment law for students",
    "site:oead.at/en family members entry residence",
    "site:oead.at/en after graduation",
    "site:oesterreich.gv.at Aufenthaltsbewilligung Student",
    "site:oesterreich.gv.at Daueraufenthalt-EU",
    "site:oesterreich.gv.at An- und Abmeldung des Wohnsitzes",
    "site:bmbwf.gv.at/Themen/HS-Uni/Studium/Zulassung Aufnahmeverfahren",
    "site:orf.beitrag.at",
    "site:willhaben.at WG-Zimmer"
]

for q in queries:
    url = f"https://lite.duckduckgo.com/lite/"
    data = urllib.parse.urlencode({'q': q}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0',
        'Content-Type': 'application/x-www-form-urlencoded'
    })
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        urls = re.findall(r'href="(http[V-Zs]?://[^"]+)"', html)
        urls = [u for u in urls if 'duckduckgo' not in u and 'w3.org' not in u]
        print(f"--- {q} ---")
        domain = q.split(' ')[0].replace('site:', '').split('/')[0] if 'site:' in q else ''
        count = 0
        for u in urls:
            if domain in u or not domain:
                print(u)
                count += 1
                if count >= 3:
                    break
    except Exception as e:
        print(f"Error on {q}: {e}")
