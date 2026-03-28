"""Parent-Child Chunking System.

Implements hierarchical chunking where:
- Child chunks (small, 200-400 chars) are used for retrieval (high precision)
- Parent chunks (large, 800-1600 chars) provide context (high recall)

This approach combines the precision of small chunks with the context richness
of large chunks, preventing loss of important context while maintaining
accurate retrieval.

Based on RAGFlow's approach and the research on hierarchical retrieval.
"""

import hashlib
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkPair:
    """A parent-child chunk pair."""
    parent_content: str
    parent_metadata: dict
    children_content: List[str]
    children_metadata: List[dict]
    parent_id: str


def _generate_parent_id(content: str, metadata: dict) -> str:
    """Generate a stable ID for a parent chunk."""
    source = metadata.get("source", "")
    section = metadata.get("section_path", "")
    # Create hash from source + section + first 100 chars of content
    identifier = f"{source}:{section}:{content[:100]}"
    return hashlib.md5(identifier.encode()).hexdigest()[:16]


def _split_into_children(
    parent_content: str,
    min_child_size: int = 200,
    max_child_size: int = 400,
) -> List[str]:
    """Split parent chunk into child chunks.

    Splits by:
    1. Double newlines (paragraphs)
    2. Single newlines if paragraphs too large
    3. Sentences if still too large

    Args:
        parent_content: The parent chunk text
        min_child_size: Minimum child size in chars
        max_child_size: Maximum child size in chars

    Returns:
        List of child chunk strings
    """
    # First try splitting by paragraphs
    paragraphs = parent_content.split("\n\n")

    children = []
    current_child = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If paragraph is small enough, it becomes a child
        if len(para) <= max_child_size:
            if current_child and len(current_child) + len(para) + 2 <= max_child_size:
                # Combine with previous small chunk
                current_child += "\n\n" + para
            else:
                # Save previous and start new
                if current_child and len(current_child) >= min_child_size:
                    children.append(current_child)
                current_child = para
        else:
            # Paragraph is too large, need to split further
            if current_child and len(current_child) >= min_child_size:
                children.append(current_child)
                current_child = ""

            # Split large paragraph by sentences
            sentences = para.replace(". ", ".\n").split("\n")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_child) + len(sentence) + 1 <= max_child_size:
                    current_child += (" " if current_child else "") + sentence
                else:
                    if current_child:
                        children.append(current_child)
                    current_child = sentence

    # Don't forget the last child
    if current_child and len(current_child) >= min_child_size:
        children.append(current_child)

    return children


def create_parent_child_chunks(
    content: str,
    metadata: dict,
    parent_min_size: int = 800,
    parent_max_size: int = 1600,
    child_min_size: int = 200,
    child_max_size: int = 400,
) -> List[ChunkPair]:
    """Create parent-child chunk pairs from content.

    Args:
        content: Text content to chunk
        metadata: Metadata dict for the content
        parent_min_size: Minimum parent chunk size
        parent_max_size: Maximum parent chunk size
        child_min_size: Minimum child chunk size
        child_max_size: Maximum child chunk size

    Returns:
        List of ChunkPair objects
    """
    # If content is small enough, create a single parent with one child
    if len(content) <= parent_max_size:
        parent_id = _generate_parent_id(content, metadata)

        # Split into children
        children = _split_into_children(content, child_min_size, child_max_size)

        # If no valid children were created (content too small), use the whole content
        if not children:
            children = [content]

        children_metadata = []
        for i, child_content in enumerate(children):
            child_meta = dict(metadata)
            child_meta["parent_id"] = parent_id
            child_meta["child_index"] = i
            child_meta["is_child"] = True
            children_metadata.append(child_meta)

        parent_meta = dict(metadata)
        parent_meta["parent_id"] = parent_id
        parent_meta["is_parent"] = True
        parent_meta["num_children"] = len(children)

        return [ChunkPair(
            parent_content=content,
            parent_metadata=parent_meta,
            children_content=children,
            children_metadata=children_metadata,
            parent_id=parent_id,
        )]

    # Content is large, split into multiple parents
    # Split by major boundaries (double newline)
    sections = content.split("\n\n")

    chunk_pairs = []
    current_parent = ""
    current_sections = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Check if adding this section would exceed parent_max_size
        if current_parent and len(current_parent) + len(section) + 2 > parent_max_size:
            # Finalize current parent
            if len(current_parent) >= parent_min_size:
                parent_metadata = dict(metadata)
                parent_id = _generate_parent_id(current_parent, parent_metadata)

                children = _split_into_children(current_parent, child_min_size, child_max_size)
                children_metadata = []
                for i, child_content in enumerate(children):
                    child_meta = dict(metadata)
                    child_meta["parent_id"] = parent_id
                    child_meta["child_index"] = i
                    child_meta["is_child"] = True
                    children_metadata.append(child_meta)

                parent_meta = dict(parent_metadata)
                parent_meta["parent_id"] = parent_id
                parent_meta["is_parent"] = True
                parent_meta["num_children"] = len(children)

                chunk_pairs.append(ChunkPair(
                    parent_content=current_parent,
                    parent_metadata=parent_meta,
                    children_content=children,
                    children_metadata=children_metadata,
                    parent_id=parent_id,
                ))

            # Start new parent
            current_parent = section
            current_sections = [section]
        else:
            # Add to current parent
            current_parent += "\n\n" + section if current_parent else section
            current_sections.append(section)

    # Don't forget the last parent
    if current_parent and len(current_parent) >= parent_min_size:
        parent_metadata = dict(metadata)
        parent_id = _generate_parent_id(current_parent, parent_metadata)

        children = _split_into_children(current_parent, child_min_size, child_max_size)
        children_metadata = []
        for i, child_content in enumerate(children):
            child_meta = dict(metadata)
            child_meta["parent_id"] = parent_id
            child_meta["child_index"] = i
            child_meta["is_child"] = True
            children_metadata.append(child_meta)

        parent_meta = dict(parent_metadata)
        parent_meta["parent_id"] = parent_id
        parent_meta["is_parent"] = True
        parent_meta["num_children"] = len(children)

        chunk_pairs.append(ChunkPair(
            parent_content=current_parent,
            parent_metadata=parent_meta,
            children_content=children,
            children_metadata=children_metadata,
            parent_id=parent_id,
        ))

    return chunk_pairs


def chunks_to_db_format(chunk_pairs: List[ChunkPair]) -> List[dict]:
    """Convert chunk pairs to flat DB format.

    Returns list of dicts with 'content' and 'metadata' keys,
    where both parent and children are separate entries.

    Args:
        chunk_pairs: List of ChunkPair objects

    Returns:
        Flat list of chunk dicts for DB insertion
    """
    db_chunks = []

    for pair in chunk_pairs:
        # Add parent
        db_chunks.append({
            "content": pair.parent_content,
            "metadata": pair.parent_metadata,
        })

        # Add children
        for child_content, child_meta in zip(pair.children_content, pair.children_metadata):
            db_chunks.append({
                "content": child_content,
                "metadata": child_meta,
            })

    return db_chunks
