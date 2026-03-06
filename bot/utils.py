import re
from typing import List

from crag.simple_rag import Document

# Tags that Telegram's HTML ParseMode actually supports
_TG_ALLOWED_TAGS = frozenset({
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "a", "code", "pre", "blockquote", "tg-spoiler", "span",
})

# Matches any HTML tag (opening, closing, self-closing)
_TAG_RE = re.compile(r"<(/?)(\w[\w-]*)((?:\s+[^>]*)?)/?>", re.IGNORECASE)


def sanitize_telegram_html(text: str) -> str:
    """Strip all HTML tags that Telegram does not support.

    Keeps only the tags listed in _TG_ALLOWED_TAGS.
    Self-closing tags like <br/> are replaced with newlines.
    Block-level tags like <p>, <div>, <li>, <details> are replaced
    with newlines so the text layout remains readable.
    """
    # First pass: replace block-level tags with newlines
    block_tags = re.compile(
        r"</?(?:p|div|li|ul|ol|details|summary|section|article|header|footer|h[1-6]|tr|td|th|table|thead|tbody)"
        r"(?:\s+[^>]*)?>",
        re.IGNORECASE,
    )
    text = block_tags.sub("\n", text)

    # Replace <br> variants with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Second pass: strip remaining unsupported inline tags
    def _replace(m: re.Match) -> str:
        tag_name = m.group(2).lower()
        if tag_name in _TG_ALLOWED_TAGS:
            return m.group(0)  # keep as-is
        return ""  # strip

    text = _TAG_RE.sub(_replace, text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def make_html_link(url: str, name: str) -> str:
    return f"<a href='{url}'>{name}</a>"


def make_html_quote(text: str) -> str:
    return f"<blockquote>{text}</blockquote>"


def tg_message_to_source_str(doc: Document) -> str:
    author = doc.metadata.get("author", "Ассистент")
    title = doc.metadata.get("title", doc.metadata.get("source", "Документ"))

    if doc.metadata.get("is_public"):
        link = doc.metadata.get("source", "")
        link_str = make_html_link(link, title)
        return f"{link_str} от {author}"
    else:
        doc_quote = make_html_quote(doc.page_content)
        return f"{title} от {author}: {doc_quote}"


def web_doc_to_source_str(doc: Document) -> str:
    title = doc.metadata.get("title", doc.metadata.get("source", "Документ"))
    section = doc.metadata.get("section_path", "")
    if section:
        title = f"{title} — {section}"
    # Prefer source_url (real URL) over source (filename)
    link = doc.metadata.get("source_url", doc.metadata.get("source", ""))
    if link and link.startswith("http"):
        return make_html_link(link, title)
    return title


def docs_to_sources_str(documents: List[Document]) -> str:
    seen_urls = set()
    source_text_rows = []
    idx = 1
    for doc in documents:
        if "is_public" in doc.metadata:
            link = doc.metadata["source"]
            if link != "":
                if link not in seen_urls:
                    source_str = tg_message_to_source_str(doc)
                    seen_urls.add(link)
                else:
                    continue
            else:
                source_str = tg_message_to_source_str(doc)
        else:
            # Deduplicate by source_url (real URL) or by source (filename)
            dedup_key = doc.metadata.get("source_url", doc.metadata.get("source", ""))
            if dedup_key in seen_urls:
                continue
            seen_urls.add(dedup_key)
            source_str = web_doc_to_source_str(doc)

        out_row = f"[{idx}] {source_str}"
        source_text_rows.append(out_row)
        idx += 1

    sources_text = "\n".join(source_text_rows)
    return sources_text


def remove_bot_command(message: str, command: str, bot_name: str):
    return message.removeprefix(f"/{command}").removeprefix(bot_name).strip()
