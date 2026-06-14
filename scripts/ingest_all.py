#!/usr/bin/env python3
"""
Austria Admission Bot — Knowledge Base Scraper v3.0
Run: python ingest_all.py

Changes in v3.0:
- Markdown headers preserved from HTML headings
- related_docs field in frontmatter for cross-referencing
- Deduplication of repeated text blocks
- Schema validation for source entries
"""

import httpx
import re
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date
import time
import hashlib
import json
import yaml

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AustriaAdmissionBot/1.0; research purposes)",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8"
}

SOURCES = [
    # ── studyinaustria.at ──────────────────────────────────────────
    {
        "url": "https://studyinaustria.at/en/plan-your-studies/application-and-admission",
        "file": "admission/application-and-admission-general.md",
        "topic": "general-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "legal/student-visa-entry.md",
            "financial/studiengebuehren.md",
            "language/german-requirements.md"
        ]
    },
    {
        "url": "https://studyinaustria.at/en/live-and-work/entry-and-visa",
        "file": "legal/student-visa-entry.md",
        "topic": "visa", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "legal/aufenthaltstitel-student.md",
            "legal/aufenthaltsbewilligung-student-oead.md",
            "admission/application-and-admission-general.md"
        ]
    },
    {
        "url": "https://studyinaustria.at/en/plan-your-studies/tuition-fee",
        "file": "financial/studiengebuehren.md",
        "topic": "tuition-fees", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "financial/stipendien.md",
            "admission/application-and-admission-general.md"
        ]
    },
    {
        "url": "https://studyinaustria.at/en/plan-your-studies/scholarships-funding",
        "file": "financial/stipendien.md",
        "topic": "scholarships", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "financial/studiengebuehren.md"
        ]
    },
    {
        "url": "https://studyinaustria.at/en/plan-your-studies/learning-german",
        "file": "language/german-requirements.md",
        "topic": "german-language", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "admission/application-and-admission-general.md"
        ]
    },
    {
        "url": "https://studyinaustria.at/en/study/information-for-students-and-researchers-from-ukraine",
        "file": "countries/ukraine-special-rules.md",
        "topic": "ukraine-special", "country_scope": ["UA"], "level": ["bachelor", "master"],
        "related_docs": [
            "countries/ukraine-faq-bmfwf.md",
            "legal/student-visa-entry.md",
            "nostrification/enic-naric-general.md"
        ]
    },
    # ── oesterreich.gv.at ─────────────────────────────────────────
    {
        "url": "https://www.oesterreich.gv.at/en/themen/bildung_und_ausbildung/hochschulen/universitaet/5/zulassung-als-ordentliche-studierende-.html",
        "file": "admission/zulassung-ordentliche-studierende.md",
        "topic": "general-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "admission/application-and-admission-general.md"
        ]
    },
    # ── oead.at / ENIC NARIC ──────────────────────────────────────
    {
        "url": "https://oead.at/en/study-research-teaching/coming-to-austria-information-and-services/recognition-and-certifications-enic-naric-austria/",
        "file": "nostrification/enic-naric-general.md",
        "topic": "nostrification", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "nostrification/school-certificate-bmb.md",
            "nostrification/university-degree-univie.md"
        ]
    },
    # ── bmfwf.gv.at / Ukraine ────────────────────────────────────
    {
        "url": "https://www.bmfwf.gv.at/en/science/ukraine/FAQs.html",
        "file": "countries/ukraine-faq-bmfwf.md",
        "topic": "ukraine-special", "country_scope": ["UA"], "level": ["bachelor", "master"],
        "related_docs": [
            "countries/ukraine-special-rules.md",
            "nostrification/enic-naric-general.md"
        ]
    },
    # ── bmb.gv.at / Nostrification ───────────────────────────────
    {
        "url": "https://www.bmb.gv.at/Themen/schule/schulrecht/anauschubi/nostr.html",
        "file": "nostrification/school-certificate-bmb.md",
        "topic": "nostrification", "country_scope": ["ALL"], "level": ["school"],
        "related_docs": [
            "nostrification/enic-naric-general.md",
            "nostrification/university-degree-univie.md"
        ]
    },
    # ── Universities ─────────────────────────────────────────────
    {
        "url": "https://www.uibk.ac.at/studium/anmeldung-zulassung/index.html.en",
        "file": "universities/uni-innsbruck/admission-general.md",
        "topic": "university-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "university": "uni-innsbruck",
        "related_docs": [
            "admission/application-and-admission-general.md",
            "nostrification/enic-naric-general.md",
            "legal/student-visa-entry.md"
        ]
    },
    {
        "url": "https://studieren.univie.ac.at/en/admission/bachelordiploma-programmes/zulassung/non-eueea-with-little-german/",
        "file": "universities/uni-wien/bachelor-non-eu-admission.md",
        "topic": "university-admission", "country_scope": ["RU", "UA", "BY", "KZ", "ALL"], "level": ["bachelor"],
        "university": "uni-wien",
        "related_docs": [
            "language/german-requirements.md",
            "nostrification/school-certificate-bmb.md",
            "legal/student-visa-entry.md",
            "financial/studiengebuehren.md"
        ]
    },
    {
        "url": "https://www.wu.ac.at/en/programs/application-and-admission/legalize/full",
        "file": "universities/wu-wien/legalization-documents.md",
        "topic": "university-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "university": "wu-wien",
        "related_docs": [
            "admission/application-and-admission-general.md",
            "nostrification/enic-naric-general.md"
        ]
    },
    # ── Страны ────────────────────────────────────────────────────
    {
        "url": "https://studieren.univie.ac.at/en/admission/bachelordiploma-programmes/zulassung/non-eueea/",
        "file": "countries/russia-belarus-non-eu.md",
        "topic": "general-admission", "country_scope": ["RU", "BY"], "level": ["bachelor"],
        "related_docs": [
            "admission/application-and-admission-general.md",
            "legal/student-visa-entry.md",
            "nostrification/school-certificate-bmb.md"
        ]
    },
    # ── Legal / ВНЖ ───────────────────────────────────────────────
    {
        "url": "https://www.bmi.gv.at/312_EN/33/start.aspx",
        "file": "legal/aufenthaltstitel-student.md",
        "topic": "visa", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "legal/student-visa-entry.md",
            "legal/aufenthaltsbewilligung-student-oead.md"
        ]
    },
    {
        "url": "https://oead.at/de/nach-oesterreich/einreise-und-aufenthalt/aufenthaltsbewilligung-student-teilnahme-mobilitaetsprogramm",
        "file": "legal/aufenthaltsbewilligung-student-oead.md",
        "topic": "visa", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "related_docs": [
            "legal/student-visa-entry.md",
            "legal/aufenthaltstitel-student.md"
        ]
    },
    # ── Нострификация диплома ─────────────────────────────────────
    {
        "url": "https://studienpraeses.univie.ac.at/en/nostrification/",
        "file": "nostrification/university-degree-univie.md",
        "topic": "nostrification", "country_scope": ["ALL"], "level": ["master"],
        "related_docs": [
            "nostrification/enic-naric-general.md",
            "nostrification/school-certificate-bmb.md"
        ]
    },
    # ── TU Wien ───────────────────────────────────────────────────
    {
        "url": "https://www.tuwien.at/en/studies/admission",
        "file": "universities/tu-wien/admission-general.md",
        "topic": "university-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "university": "tu-wien",
        "related_docs": [
            "universities/tu-wien/bachelor-international.md",
            "universities/tu-wien/master-international.md",
            "admission/application-and-admission-general.md"
        ]
    },
    {
        "url": "https://www.tuwien.at/en/studies/admission/bachelors-programmes/admission-with-an-international-school-leaving-certificate",
        "file": "universities/tu-wien/bachelor-international.md",
        "topic": "university-admission", "country_scope": ["RU", "UA", "BY", "KZ", "ALL"], "level": ["bachelor"],
        "university": "tu-wien",
        "related_docs": [
            "universities/tu-wien/admission-general.md",
            "nostrification/school-certificate-bmb.md",
            "legal/student-visa-entry.md",
            "financial/studiengebuehren.md",
            "language/german-requirements.md"
        ]
    },
    {
        "url": "https://www.tuwien.at/en/studies/admission/masters-programmes/other-inter-national-bachelors-degree",
        "file": "universities/tu-wien/master-international.md",
        "topic": "university-admission", "country_scope": ["RU", "UA", "BY", "KZ", "ALL"], "level": ["master"],
        "university": "tu-wien",
        "related_docs": [
            "universities/tu-wien/admission-general.md",
            "nostrification/university-degree-univie.md",
            "legal/student-visa-entry.md",
            "financial/studiengebuehren.md"
        ]
    },
    # ── MedUni Wien ───────────────────────────────────────────────
    {
        "url": "https://www.meduniwien.ac.at/web/en/studies-further-education/application-admission/",
        "file": "universities/meduni-wien/admission-general.md",
        "topic": "university-admission", "country_scope": ["ALL"], "level": ["bachelor"],
        "university": "meduni-wien",
        "js_render": True,
        "related_docs": [
            "admission/application-and-admission-general.md",
            "legal/student-visa-entry.md"
        ]
    },
    # ── Uni Graz ──────────────────────────────────────────────────
    {
        "url": "https://studyinaustria.at/en/study/institutions/universities/university-of-graz",
        "file": "universities/uni-graz/admission-general.md",
        "topic": "university-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "university": "uni-graz",
        "related_docs": [
            "admission/application-and-admission-general.md",
            "nostrification/enic-naric-general.md",
            "legal/student-visa-entry.md"
        ]
    },
    # ── Uni Salzburg ──────────────────────────────────────────────
    {
        "url": "https://www.plus.ac.at/studium/vor-dem-studium-3/zulassungsvoraussetzungen/?lang=en",
        "file": "universities/uni-salzburg/admission-general.md",
        "topic": "university-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "university": "uni-salzburg",
        "js_render": True,
        "related_docs": [
            "admission/application-and-admission-general.md",
            "nostrification/enic-naric-general.md"
        ]
    },
    {
        "url": "https://www.plus.ac.at/studium/vor-dem-studium-3/zulassungsvoraussetzungen/information-fuer-internationale-studieninteressierte/?lang=en",
        "file": "universities/uni-salzburg/international-students.md",
        "topic": "university-admission", "country_scope": ["RU", "UA", "BY", "KZ", "ALL"], "level": ["bachelor", "master"],
        "university": "uni-salzburg",
        "js_render": True,
        "related_docs": [
            "universities/uni-salzburg/admission-general.md",
            "legal/student-visa-entry.md",
            "language/german-requirements.md"
        ]
    },
    # ── JKU Linz ─────────────────────────────────────────────────
    {
        "url": "https://www.jku.at/en/teaching-and-studies-organization/admissions-office/",
        "file": "universities/jku-linz/admission-general.md",
        "topic": "university-admission", "country_scope": ["ALL"], "level": ["bachelor", "master"],
        "university": "jku-linz",
        "related_docs": [
            "admission/application-and-admission-general.md",
            "nostrification/enic-naric-general.md",
            "legal/student-visa-entry.md"
        ]
    },
]

IGNORE_TAGS = ["header", "footer", "nav", "aside", "script", "style", "noscript", "iframe"]
IGNORE_CLASSES_RE = re.compile(
    r"cookie|privacy|language.?switch|breadcrumb|skip|sidebar|social|share|print|ad-|banner", re.I
)
MIN_CONTENT_LENGTH = 400
REQUEST_DELAY = 1.5  # секунд между запросами


def load_schema(schema_path: str = "knowledge_base/schema.yaml") -> dict:
    """Load and return the knowledge base schema for validation."""
    path = Path(schema_path)
    if not path.exists():
        print(f"⚠️  Schema file not found at {schema_path}, skipping validation")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_sources(sources: list, schema: dict) -> list[str]:
    """Validate all source entries against the schema. Returns list of errors."""
    if not schema:
        return []

    errors = []
    valid_topics = set(schema.get("topics", []))
    valid_universities = set(schema.get("universities", []))
    valid_scopes = set(schema.get("country_scopes", []))
    valid_levels = set(schema.get("levels", []))

    for i, src in enumerate(sources):
        label = src.get("file", f"source #{i}")

        if src["topic"] not in valid_topics:
            errors.append(f"[{label}] Invalid topic: '{src['topic']}'. Allowed: {valid_topics}")

        for scope in src.get("country_scope", []):
            if scope not in valid_scopes:
                errors.append(f"[{label}] Invalid country_scope: '{scope}'. Allowed: {valid_scopes}")

        for level in src.get("level", []):
            if level not in valid_levels:
                errors.append(f"[{label}] Invalid level: '{level}'. Allowed: {valid_levels}")

        univ = src.get("university")
        if univ and univ not in valid_universities:
            errors.append(f"[{label}] Invalid university: '{univ}'. Allowed: {valid_universities}")

    return errors


def clean_html(soup: BeautifulSoup) -> str:
    """Extract text from HTML, preserving headings as Markdown headers."""
    for tag in IGNORE_TAGS:
        for el in soup.find_all(tag):
            el.decompose()
    for el in soup.find_all(class_=IGNORE_CLASSES_RE):
        el.decompose()

    main = (
        soup.find("main") or
        soup.find(id="content") or
        soup.find(id=re.compile(r"main", re.I)) or
        soup.find(class_=re.compile(r"^(main|content|article)$", re.I)) or
        soup.find("article") or
        soup.body
    )

    if not main:
        return ""

    # Convert HTML headings to Markdown headers before extracting text
    for level in range(1, 7):
        for tag in main.find_all(f"h{level}"):
            heading_text = tag.get_text(strip=True)
            if heading_text:
                tag.string = f"\n{'#' * level} {heading_text}\n"

    # Convert list items to Markdown bullets
    for li in main.find_all("li"):
        li_text = li.get_text(strip=True)
        if li_text:
            li.string = f"- {li_text}"

    return main.get_text(separator="\n", strip=True)


def text_to_clean_md(text: str) -> str:
    lines = text.split("\n")
    cleaned, prev_empty = [], False
    for line in lines:
        line = line.strip()
        if not line:
            if not prev_empty:
                cleaned.append("")
            prev_empty = True
        else:
            cleaned.append(line)
            prev_empty = False
    return "\n".join(cleaned).strip()


def deduplicate_blocks(text: str, min_block_lines: int = 4) -> str:
    """Remove duplicate multi-line blocks that appear more than once.
    
    This catches repeated sections like contact info, office hours, etc.
    that appear multiple times on the same page.
    """
    lines = text.split("\n")
    if len(lines) < min_block_lines * 2:
        return text

    # Build blocks of min_block_lines consecutive non-empty lines
    seen_blocks = set()
    lines_to_remove = set()

    for i in range(len(lines) - min_block_lines + 1):
        block = tuple(lines[i:i + min_block_lines])
        # Skip blocks that are mostly empty
        non_empty = sum(1 for l in block if l.strip())
        if non_empty < min_block_lines - 1:
            continue

        block_key = "\n".join(block).strip()
        if not block_key:
            continue

        if block_key in seen_blocks:
            # Mark all lines of this duplicate block for removal
            # Also extend to remove subsequent lines that are part of the same
            # repeated section
            j = i
            while j < len(lines):
                remaining_block = tuple(lines[j:j + min_block_lines])
                remaining_key = "\n".join(remaining_block).strip()
                if remaining_key in seen_blocks:
                    for k in range(j, min(j + min_block_lines, len(lines))):
                        lines_to_remove.add(k)
                    j += min_block_lines
                else:
                    break
        else:
            seen_blocks.add(block_key)

    if not lines_to_remove:
        return text

    result = [line for i, line in enumerate(lines) if i not in lines_to_remove]
    return "\n".join(result).strip()


def postprocess(text: str) -> str:
    lines = text.split("\n")
    # Убираем немецкие хвосты (артефакт bilingual страниц)
    cleaned = []
    for line in lines:
        # Строка < 80/100 символов и заканчивается немецким артефактом — дроп
        if re.match(r'^[a-zäöüß\s\.]+\.$', line.strip(), re.I) and len(line.strip()) < 100:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def make_frontmatter(source: dict, content_hash: str) -> str:
    today = date.today().isoformat()
    scope = json.dumps(source["country_scope"])
    level = json.dumps(source["level"])
    univ = source.get("university", None)
    related = source.get("related_docs", [])

    fm = (
        f"---\n"
        f"source_url: {source['url']}\n"
        f"last_updated: '{today}'\n"
        f"country_scope: {scope}\n"
        f"topic: {source['topic']}\n"
        f"level: {level}\n"
        f"university: {univ}\n"
        f"content_hash: {content_hash}\n"
        f"language: en\n"
        f"auto_update: true\n"
    )

    if related:
        fm += "related_docs:\n"
        for doc in related:
            fm += f"  - {doc}\n"

    fm += "---\n\n"
    return fm


def is_valid(text: str) -> tuple[bool, str]:
    if "Error 404" in text or "Document not found" in text:
        return False, "404_error"
    if "Jump to main content" in text and len(text) < MIN_CONTENT_LENGTH:
        return False, "navigation_only"
    if len(text.strip()) < MIN_CONTENT_LENGTH:
        return False, f"too_short ({len(text.strip())} chars)"
    return True, "ok"


def run():
    base_path = Path("knowledge_base")
    registry = []

    # ── Schema validation ──────────────────────────────────────────
    schema = load_schema()
    validation_errors = validate_sources(SOURCES, schema)
    if validation_errors:
        print("❌ Schema validation errors:")
        for err in validation_errors:
            print(f"   {err}")
        print("\nFix the errors above before running the scraper.")
        return

    if schema:
        print("✅ Schema validation passed")
    print()

    with httpx.Client(headers=HEADERS, timeout=25, follow_redirects=True) as client:
        for i, src in enumerate(SOURCES):
            print(f"[{i+1}/{len(SOURCES)}] Scraping: {src['url']}")
            try:
                if src.get("js_render"):
                    from playwright.sync_api import sync_playwright
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        context = browser.new_context(
                            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                            viewport={'width': 1920, 'height': 1080}
                        )
                        page = context.new_page()
                        page.goto(src["url"], wait_until="networkidle", timeout=30000)
                        html = page.content()
                        browser.close()
                else:
                    r = client.get(src["url"])
                    r.raise_for_status()
                    html = r.text

                soup = BeautifulSoup(html, "html.parser")
                raw_text = clean_html(soup)
                clean_text = text_to_clean_md(raw_text)
                deduped_text = deduplicate_blocks(clean_text)
                final_text = postprocess(deduped_text)
                valid, reason = is_valid(final_text)

                if valid:
                    content_hash = hashlib.md5(final_text.encode()).hexdigest()[:8]
                    out_path = base_path / src["file"]
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(make_frontmatter(src, content_hash) + final_text, encoding="utf-8")
                    print(f"  ✅  Saved → {src['file']} ({len(final_text)} chars, hash: {content_hash})")
                    registry.append({
                        "file": src["file"], "url": src["url"],
                        "status": "✅", "hash": content_hash,
                        "last_checked": date.today().isoformat()
                    })
                else:
                    print(f"  ⚠️   Skipped — {reason}")
                    registry.append({
                        "file": src["file"], "url": src["url"],
                        "status": f"⚠️ {reason}", "hash": "",
                        "last_checked": date.today().isoformat()
                    })

            except Exception as e:
                print(f"  💥  Error — {e}")
                registry.append({
                    "file": src["file"], "url": src["url"],
                    "status": f"💥 {str(e)[:80]}", "hash": "",
                    "last_checked": date.today().isoformat()
                })

            if i < len(SOURCES) - 1:
                time.sleep(REQUEST_DELAY)

    # Обновляем реестр
    registry_path = base_path / "_sources_registry.md"
    lines = ["# Sources Registry\n", f"*Last updated: {date.today().isoformat()}*\n\n",
             "| File | Source URL | Status | Hash | Last Checked |\n",
             "|---|---|---|---|---|\n"]
    for r in registry:
        lines.append(f"| {r['file']} | {r['url']} | {r['status']} | {r['hash']} | {r['last_checked']} |\n")
    registry_path.write_text("".join(lines), encoding="utf-8")

    ok = sum(1 for r in registry if r["status"] == "✅")
    print(f"\n{'='*50}")
    print(f"Done: {ok}/{len(SOURCES)} pages scraped successfully")
    print(f"Registry saved → knowledge_base/_sources_registry.md")


if __name__ == "__main__":
    run()
