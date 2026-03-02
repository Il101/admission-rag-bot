import re
from typing import List

from langchain_core.documents import Document

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
    author = doc.metadata["author"]
    title = doc.metadata["title"]

    if doc.metadata["is_public"]:
        link = doc.metadata["source"]
        link_str = make_html_link(link, title)
        return f"{link_str} от {author}"
    else:
        doc_quote = make_html_quote(doc.page_content)
        return f"{title} от {author}: {doc_quote}"


def web_doc_to_source_str(doc: Document) -> str:
    title = doc.metadata["title"]
    link = doc.metadata["source"]
    return make_html_link(link, title)


def docs_to_sources_str(documents: List[Document]) -> str:
    links = set()
    source_text_rows = []
    idx = 1
    for doc in documents:
        if "is_public" in doc.metadata:
            link = doc.metadata["source"]
            if link != "":
                if link not in links:
                    source_str = tg_message_to_source_str(doc)
                    links.add(link)
                else:
                    continue
            else:
                source_str = tg_message_to_source_str(doc)
        elif doc.metadata["source"] not in links:
            links.add(doc.metadata["source"])
            source_str = web_doc_to_source_str(doc)
        else:
            continue

        out_row = f"[{idx}] {source_str}"
        source_text_rows.append(out_row)
        idx += 1

    sources_text = "\n".join(source_text_rows)
    return sources_text


def remove_bot_command(message: str, command: str, bot_name: str):
    return message.removeprefix(f"/{command}").removeprefix(bot_name).strip()
