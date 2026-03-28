from pathlib import Path

import yaml


def test_gemini_prompt_config_loads():
    """Smoke-test: prompt config exists and has required RAG sections."""
    cfg_path = Path("configs/prompts/gemini-2.5-flash.yaml")
    assert cfg_path.exists(), "Expected prompt config file is missing"

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict)
    assert "rag_prompt" in config
    assert "messages" in config["rag_prompt"]
    assert isinstance(config["rag_prompt"]["messages"], list)
    assert len(config["rag_prompt"]["messages"]) > 0
