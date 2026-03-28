from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from crag.simple_rag import build_entity_topic_boost_sql, detect_entity_focus
from crag.yaml_facts_indexer import infer_entity_type_for_yaml_chunk


def _load_index_module():
    root = Path(__file__).resolve().parent.parent
    module_path = root / "init_scripts" / "index_knowledge_base.py"
    spec = spec_from_file_location("index_knowledge_base", module_path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_detect_entity_focus_housing_priority():
    query = "Есть ли в общежитии вегетарианское меню?"
    assert detect_entity_focus(query) == "housing"


def test_detect_entity_focus_food():
    query = "Сколько стоит Mensa и где веганское меню?"
    assert detect_entity_focus(query) == "food"


def test_build_entity_topic_boost_sql_housing():
    clause = build_entity_topic_boost_sql("housing")
    assert "case when" in clause.lower()
    assert "entity_type" in clause
    assert "topic" in clause
    assert "housing" in clause


def test_build_entity_topic_boost_sql_food():
    clause = build_entity_topic_boost_sql("food")
    assert "case when" in clause.lower()
    assert "entity_type" in clause
    assert "student-budget" in clause


def test_detect_entity_focus_no_housing_on_relocation_only():
    query = "Переезд в Австрию: виза D и Anmeldung"
    assert detect_entity_focus(query) is None


def test_yaml_entity_type_finance():
    metadata = {"fact_type": "tuition_tariff", "source": "facts/financial/tuition.yaml"}
    assert infer_entity_type_for_yaml_chunk(metadata) == "finance"


def test_yaml_entity_type_admission():
    metadata = {"fact_type": "deadline", "source": "facts/universities/tu-wien.yaml"}
    assert infer_entity_type_for_yaml_chunk(metadata) == "admission"


def test_markdown_entity_type_housing():
    mod = _load_index_module()
    metadata = {"topic": "housing", "source": "housing.md"}
    assert mod.infer_entity_type_from_metadata(metadata) == "housing"


def test_markdown_entity_type_food_from_budget():
    mod = _load_index_module()
    metadata = {"topic": "student-budget", "source": "student-budget.md"}
    assert mod.infer_entity_type_from_metadata(metadata) == "food"
