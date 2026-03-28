"""Tests for parent-child chunking system."""

import pytest
from crag.parent_child_chunking import (
    create_parent_child_chunks,
    chunks_to_db_format,
    _generate_parent_id,
    _split_into_children,
)


def test_generate_parent_id():
    """Parent ID should be consistent for same input."""
    content = "Test content"
    metadata = {"source": "test.md", "section_path": "intro"}

    id1 = _generate_parent_id(content, metadata)
    id2 = _generate_parent_id(content, metadata)

    assert id1 == id2
    assert len(id1) == 16  # MD5 hash truncated to 16 chars


def test_generate_parent_id_different_content():
    """Different content should produce different IDs."""
    metadata = {"source": "test.md"}

    id1 = _generate_parent_id("Content 1", metadata)
    id2 = _generate_parent_id("Content 2", metadata)

    assert id1 != id2


def test_split_into_children_small_content():
    """Small content should become single child."""
    content = "This is a small paragraph."
    children = _split_into_children(content, min_child_size=20, max_child_size=100)

    assert len(children) == 1
    assert children[0] == content


def test_split_into_children_multiple_paragraphs():
    """Multiple paragraphs should be split into children."""
    content = """First paragraph is quite long and should easily exceed the minimum child size.

Second paragraph also contains enough text to be a separate chunk in the system.

Third paragraph for testing splitting."""

    children = _split_into_children(content, min_child_size=50, max_child_size=100)

    assert len(children) >= 2
    assert all(len(child) >= 50 for child in children)


def test_split_into_children_large_paragraph():
    """Large single paragraph should be split by sentences."""
    content = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    children = _split_into_children(content, min_child_size=30, max_child_size=60)

    assert len(children) > 1
    assert all(len(child) <= 60 for child in children)


def test_create_parent_child_chunks_small_content():
    """Small content should create single chunk pair."""
    content = "Small content for testing."
    metadata = {"source": "test.md", "entity_type": "housing"}

    pairs = create_parent_child_chunks(content, metadata)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.parent_content == content
    assert pair.parent_metadata["is_parent"] is True
    assert pair.parent_metadata["entity_type"] == "housing"
    assert len(pair.children_content) >= 1
    assert all(meta["is_child"] for meta in pair.children_metadata)


def test_create_parent_child_chunks_large_content():
    """Large content should create multiple chunk pairs."""
    # Create content larger than parent_max_size (1600)
    content = "\n\n".join([
        "This is section " + str(i) + ". " + "Content paragraph. " * 100
        for i in range(5)
    ])

    metadata = {"source": "large.md", "entity_type": "visa"}

    pairs = create_parent_child_chunks(
        content,
        metadata,
        parent_min_size=800,
        parent_max_size=1600,
    )

    assert len(pairs) > 1
    for pair in pairs:
        assert len(pair.parent_content) >= 800
        # Allow overflow due to section-based splitting
        assert len(pair.parent_content) <= 2000
        assert pair.parent_metadata["is_parent"] is True
        assert pair.parent_metadata["entity_type"] == "visa"
        assert len(pair.children_content) >= 1


def test_create_parent_child_chunks_metadata_inheritance():
    """Children should inherit parent metadata."""
    content = "First paragraph. Another paragraph. More content here."
    metadata = {
        "source": "test.md",
        "entity_type": "finance",
        "university": "TU Wien",
    }

    pairs = create_parent_child_chunks(content, metadata)
    pair = pairs[0]

    for child_meta in pair.children_metadata:
        assert child_meta["source"] == "test.md"
        assert child_meta["entity_type"] == "finance"
        assert child_meta["university"] == "TU Wien"
        assert child_meta["parent_id"] == pair.parent_id
        assert child_meta["is_child"] is True


def test_chunks_to_db_format():
    """Conversion to DB format should create separate entries for parents and children."""
    content = "First paragraph. Second paragraph. Third paragraph."
    metadata = {"source": "test.md", "entity_type": "admission"}

    pairs = create_parent_child_chunks(content, metadata)
    db_chunks = chunks_to_db_format(pairs)

    # Should have parents + children entries
    total_entries = len(pairs) + sum(len(pair.children_content) for pair in pairs)
    assert len(db_chunks) == total_entries

    # First entry should be parent
    assert db_chunks[0]["metadata"]["is_parent"] is True

    # Remaining should be children
    for chunk in db_chunks[1:]:
        if chunk["metadata"].get("is_child"):
            assert "parent_id" in chunk["metadata"]


def test_parent_id_consistency():
    """Parent IDs should be consistent within a chunk pair."""
    content = "Test content paragraph one. Another paragraph."
    metadata = {"source": "test.md"}

    pairs = create_parent_child_chunks(content, metadata)
    pair = pairs[0]

    # All children should have same parent_id
    assert all(meta["parent_id"] == pair.parent_id for meta in pair.children_metadata)


def test_child_indexing():
    """Children should have proper indexing."""
    content = "First. Second. Third. Fourth. Fifth."
    metadata = {"source": "test.md"}

    pairs = create_parent_child_chunks(content, metadata, child_max_size=20)
    pair = pairs[0]

    for i, child_meta in enumerate(pair.children_metadata):
        assert child_meta["child_index"] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
