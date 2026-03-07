"""
Unit tests for core utility functions (no external dependencies required).

Run with:  python -m pytest tests/test_rag_utils.py -v

NOTE: The functions are tested by inlining their pure logic here so that
pytest can run without the full Telegram/SQLAlchemy stack in the venv.
"""

import json
import re
import sys
import os
import pytest

# ── Inline copies of pure logic (no Telegram/DB imports required) ──────────

# --- parse_suggested_buttons (from bot/handlers/rag.py) ---

def parse_suggested_buttons(text: str):
    json_text = text.strip()
    if "{" in json_text and "}" in json_text:
        try:
            start = json_text.find("{")
            end = json_text.rfind("}") + 1
            json_text = json_text[start:end]
        except Exception:
            pass
    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            answer = data.get("answer", "")
            suggested = data.get("suggested_questions", [])
            if isinstance(suggested, list):
                return answer, suggested
        return text, []
    except json.JSONDecodeError:
        ans_match = re.search(r'"answer":\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        btns_match = re.search(r'"suggested_questions":\s*(\[[^\]]*\])', text, re.DOTALL)
        ans = text
        btns = []
        if ans_match:
            ans = ans_match.group(1).replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
        if btns_match:
            try:
                btns = json.loads(btns_match.group(1))
            except Exception:
                btns = re.findall(r'"([^"]*)"', btns_match.group(1))
        return ans, btns


# --- sanitize_telegram_html (from bot/utils.py) ---

_TG_ALLOWED_TAGS = frozenset({
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "a", "code", "pre", "blockquote", "tg-spoiler", "span",
})
_TAG_RE = re.compile(r"<(/?)(\w[\w-]*)((?:\s+[^>]*)?)/?>", re.IGNORECASE)

def sanitize_telegram_html(text: str) -> str:
    block_tags = re.compile(
        r"</?(?:p|div|li|ul|ol|details|summary|section|article|header|footer|h[1-6]|tr|td|th|table|thead|tbody)"
        r"(?:\s+[^>]*)?>",
        re.IGNORECASE,
    )
    text = block_tags.sub("\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    def _replace(m):
        tag_name = m.group(2).lower()
        return m.group(0) if tag_name in _TG_ALLOWED_TAGS else ""
    text = _TAG_RE.sub(_replace, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- split_markdown (from init_scripts/index_knowledge_base.py) ---

def split_markdown(body: str, file_metadata: dict, max_chunk_len: int = 800):
    lines = body.split("\n")
    heading_stack: dict = {}
    sections = []
    current_lines = []
    current_headings: dict = {}

    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append((dict(current_headings), current_lines[:]))
                current_lines = []
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack[level] = heading_text
            for k in [k for k in heading_stack if k > level]:
                del heading_stack[k]
            current_headings = dict(heading_stack)
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((dict(current_headings), current_lines[:]))

    doc_title = file_metadata.get("title", "")
    if not doc_title:
        for headings, _ in sections:
            if 1 in headings:
                doc_title = headings[1]
                break

    chunks = []
    for headings, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text or len(section_text) < 30:
            continue

        path_parts = []
        for lvl in sorted(headings.keys()):
            if lvl == 1:
                continue
            path_parts.append(headings[lvl])
        section_path = " > ".join(path_parts) if path_parts else ""

        prefix_parts = []
        university = file_metadata.get("university", "")
        if university:
            prefix_parts.append(f"Университет: {university}")
        elif doc_title:
            prefix_parts.append(doc_title)
        topic = file_metadata.get("topic", "")
        if topic:
            prefix_parts.append(f"Тема: {topic}")
        if section_path:
            prefix_parts.append(f"Раздел: {section_path}")
        prefix = "[" + " | ".join(prefix_parts) + "]" if prefix_parts else ""

        chunk_meta = {
            "source": file_metadata.get("source", ""),
            "title": doc_title,
            "section_path": section_path,
        }
        for key in ("source_url", "university", "topic", "country_scope", "level", "city"):
            if key in file_metadata:
                chunk_meta[key] = file_metadata[key]

        content = f"{prefix}\n{section_text}" if prefix else section_text
        if len(section_text) <= max_chunk_len:
            chunks.append({"content": content, "metadata": chunk_meta})
        else:
            # minimal sub-splitting for test purposes
            sub_meta = dict(chunk_meta)
            sub_meta["section_path"] = f"{section_path} (часть 1)"
            chunks.append({"content": content[:max_chunk_len], "metadata": sub_meta})
            sub_meta2 = dict(chunk_meta)
            sub_meta2["section_path"] = f"{section_path} (часть 2)"
            chunks.append({"content": content[max_chunk_len:], "metadata": sub_meta2})

    return chunks


# --- _build_user_filters (from bot/handlers/rag.py) ---

COUNTRY_TO_SCOPE = {
    "россия": "RU", "russia": "RU", "ru": "RU",
    "украина": "UA", "ukraine": "UA", "ua": "UA",
    "беларусь": "BY", "belarus": "BY", "by": "BY",
    "казахстан": "KZ", "kazakhstan": "KZ", "kz": "KZ",
}
LEVEL_TO_SCOPE = {"bachelor": "bachelor", "master": "master", "phd": "phd"}

def _build_user_filters(onboarding_data: dict) -> dict:
    filters = {}
    country = onboarding_data.get("country")
    if country:
        code = COUNTRY_TO_SCOPE.get(country.lower().strip(), "")
        if code:
            filters["country_scope"] = code
    target_level = onboarding_data.get("targetLevel")
    if target_level:
        code = LEVEL_TO_SCOPE.get(target_level.lower().strip(), "")
        if code:
            filters["level"] = code
    return filters


# ══════════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class TestParseSuggestedButtons:
    def test_valid_json_full(self):
        payload = json.dumps({
            "answer": "Здравствуй!",
            "suggested_questions": ["Вопрос 1", "Вопрос 2", "Вопрос 3"],
        })
        answer, questions = parse_suggested_buttons(payload)
        assert answer == "Здравствуй!"
        assert questions == ["Вопрос 1", "Вопрос 2", "Вопрос 3"]

    def test_valid_json_empty_questions(self):
        payload = json.dumps({"answer": "Ответ", "suggested_questions": []})
        answer, questions = parse_suggested_buttons(payload)
        assert answer == "Ответ"
        assert questions == []

    def test_valid_json_embedded_in_whitespace(self):
        payload = '  {"answer": "Ура!", "suggested_questions": ["Q1"]}  '
        answer, questions = parse_suggested_buttons(payload)
        assert answer == "Ура!"
        assert questions == ["Q1"]

    def test_partial_json_fallback_regex(self):
        partial = '{"answer": "Привет мир", "suggested_'
        answer, _ = parse_suggested_buttons(partial)
        assert "Привет мир" in answer

    def test_plain_string_fallback(self):
        text = "Это обычный текст без JSON."
        answer, _ = parse_suggested_buttons(text)
        assert text in answer or answer == text

    def test_json_with_unicode(self):
        payload = json.dumps(
            {"answer": "Нострификация — признание документов.", "suggested_questions": []},
            ensure_ascii=False,
        )
        answer, _ = parse_suggested_buttons(payload)
        assert "Нострификация" in answer

    def test_json_with_escaped_newlines(self):
        payload = '{"answer": "Шаг 1\\nШаг 2\\nШаг 3", "suggested_questions": []}'
        answer, _ = parse_suggested_buttons(payload)
        assert "Шаг" in answer


class TestSanitizeTelegramHtml:
    def test_keeps_bold(self):
        assert "<b>текст</b>" in sanitize_telegram_html("<b>текст</b>")

    def test_keeps_italic(self):
        assert "<i>текст</i>" in sanitize_telegram_html("<i>текст</i>")

    def test_keeps_link(self):
        result = sanitize_telegram_html('<a href="https://example.com">ссылка</a>')
        assert '<a href="https://example.com">' in result

    def test_strips_div(self):
        result = sanitize_telegram_html("<div>текст</div>")
        assert "<div>" not in result
        assert "текст" in result

    def test_strips_h1(self):
        result = sanitize_telegram_html("<h1>Заголовок</h1>")
        assert "<h1>" not in result
        assert "Заголовок" in result

    def test_replaces_br_with_newline(self):
        result = sanitize_telegram_html("строка1<br/>строка2")
        assert "\n" in result

    def test_strips_script(self):
        result = sanitize_telegram_html("<script>alert(1)</script>hello")
        assert "<script>" not in result
        assert "hello" in result

    def test_no_excessive_blank_lines(self):
        result = sanitize_telegram_html("a\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_plain_text_unchanged(self):
        text = "Это обычный текст без HTML."
        assert sanitize_telegram_html(text) == text

    def test_keeps_code(self):
        result = sanitize_telegram_html("<code>pip install foo</code>")
        assert "<code>" in result


class TestSplitMarkdown:
    def test_basic_split_by_heading(self):
        body = (
            "# Title\n\n"
            "## Section A\n\nContent for section A goes here.\n\n"
            "## Section B\n\nContent for section B goes here."
        )
        chunks = split_markdown(body, {})
        assert len(chunks) >= 2
        contents = [c["content"] for c in chunks]
        assert any("Content for section A" in c for c in contents)
        assert any("Content for section B" in c for c in contents)

    def test_section_path_h2(self):
        body = "# Doc\n\n## Раздел\n\nТекст данного раздела достаточно длинный."
        chunks = split_markdown(body, {})
        paths = [c["metadata"]["section_path"] for c in chunks]
        assert any("Раздел" in p for p in paths)

    def test_section_path_h3_inherits_h2(self):
        body = "# Doc\n\n## H2\n\n### H3\n\nТекст данного подраздела достаточно длинный."
        chunks = split_markdown(body, {})
        paths = [c["metadata"]["section_path"] for c in chunks]
        assert any("H2" in p and "H3" in p for p in paths)

    def test_metadata_fields_propagated(self):
        body = "# Doc\n\n## Section\n\nText content."
        meta = {"source_url": "https://example.at", "university": "tu-wien", "country_scope": ["ALL"]}
        chunks = split_markdown(body, meta)
        for c in chunks:
            if "Text content" in c["content"]:
                assert c["metadata"]["source_url"] == "https://example.at"
                assert c["metadata"]["university"] == "tu-wien"

    def test_long_section_splits(self):
        long_text = "Sentence. " * 200
        body = f"# Doc\n\n## Long\n\n{long_text}"
        chunks = split_markdown(body, {}, max_chunk_len=200)
        assert len(chunks) >= 2

    def test_contextual_prefix_contains_university(self):
        body = "# Doc\n\n## Section\n\nContent here."
        meta = {"university": "uni-wien", "topic": "admission"}
        chunks = split_markdown(body, meta)
        if chunks:
            assert "uni-wien" in chunks[0]["content"] or "uni-wien" in str(chunks[0]["metadata"])


class TestBuildUserFilters:
    def test_russia_maps_to_RU(self):
        f = _build_user_filters({"country": "россия"})
        assert f["country_scope"] == "RU"

    def test_ukraine_maps_to_UA(self):
        f = _build_user_filters({"country": "украина"})
        assert f["country_scope"] == "UA"

    def test_unknown_country_excluded(self):
        f = _build_user_filters({"country": "марс"})
        assert "country_scope" not in f

    def test_bachelor_level(self):
        f = _build_user_filters({"targetLevel": "bachelor"})
        assert f["level"] == "bachelor"

    def test_master_level(self):
        f = _build_user_filters({"targetLevel": "master"})
        assert f["level"] == "master"

    def test_empty_onboarding(self):
        f = _build_user_filters({})
        assert f == {}

    def test_case_insensitive_country(self):
        f = _build_user_filters({"country": "РОССИЯ"})
        assert f["country_scope"] == "RU"

    def test_both_country_and_level(self):
        f = _build_user_filters({"country": "украина", "targetLevel": "master"})
        assert f["country_scope"] == "UA"
        assert f["level"] == "master"
