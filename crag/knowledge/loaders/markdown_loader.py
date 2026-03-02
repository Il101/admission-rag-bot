import os
import yaml
from typing import List
from pathlib import Path
from langchain_core.documents import Document

def _parse_frontmatter(content: str):
    """Splits YAML frontmatter from Markdown body"""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1])
                body = parts[2].strip()
                return metadata, body
            except Exception as e:
                print(f"Failed to parse frontmatter: {e}")
    return {}, content

def load(directory_path: str = "knowledge_base") -> List[Document]:
    """
    Loads all Markdown files from the directory.
    Extracts YAML Frontmatter into LangChain Document.metadata.
    """
    docs = []
    base_path = Path(directory_path)
    
    if not base_path.exists():
        print(f"Error: Directory {directory_path} not found.")
        return docs
        
    for filepath in base_path.rglob("*.md"):
        # Ignore registry index
        if filepath.name.startswith("_"):
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        metadata, body = _parse_frontmatter(content)
        
        # Add basic file info to metadata
        metadata["source"] = str(filepath.relative_to(base_path))
        metadata["title"] = filepath.stem.replace("-", " ").title()
        
        doc = Document(
            page_content=body,
            metadata=metadata
        )
        docs.append(doc)
        
    print(f"Successfully loaded {len(docs)} documents from {directory_path}")
    return docs
