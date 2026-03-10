"""
RAG Evaluation Script — measures retrieval quality and answer faithfulness.

Metrics:
  - Retrieval Recall@K: % of expected keywords found in retrieved docs
  - Answer Keyword Coverage: % of expected keywords present in the generated answer
  - Hallucination Rate: % of answers that contain info not grounded in retrieved context
  - Average Latency: time per question (rewrite + retrieve + grade + generate)

Usage:
    python tests/eval_rag.py [--dataset tests/eval_dataset.json] [--top-k 6] [--output outputs/eval_report.json]
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot.env  # noqa: F401 — loads .env
import yaml
from sqlalchemy import create_engine

from crag.simple_rag import SimpleRAG, Document, documents_to_context_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    """Load Hydra config manually (no @hydra.main decorator)."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve env vars in bot_db_connection
    db_url = config.get("bot_db_connection", "")
    db_url = (
        db_url.replace("${oc.env:POSTGRES_USER}", os.environ.get("POSTGRES_USER", ""))
        .replace("${oc.env:POSTGRES_PASSWORD}", os.environ.get("POSTGRES_PASSWORD", ""))
        .replace("${oc.env:POSTGRES_HOST}", os.environ.get("POSTGRES_HOST", ""))
        .replace("${oc.env:POSTGRES_DB}", os.environ.get("POSTGRES_DB", ""))
    )

    # Load prompts
    prompts_key = config.get("prompts", {})
    if isinstance(prompts_key, str):
        # If it's a reference, load the file directly
        prompts_path = Path(__file__).resolve().parent.parent / "configs" / "prompts" / "gemini-2.5-flash.yaml"
        with open(prompts_path, "r") as f:
            prompts = yaml.safe_load(f)
    elif isinstance(prompts_key, dict):
        prompts = prompts_key
    else:
        prompts = {}

    return db_url, prompts


def keyword_recall(expected_keywords: list[str], text: str) -> float:
    """Fraction of expected keywords found in text (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return found / len(expected_keywords)


async def evaluate_single(
    simple_rag: SimpleRAG,
    item: dict,
    top_k: int = 6,
) -> dict:
    """Evaluate a single question from the dataset."""
    question = item["question"]
    expected_kw = item.get("expected_keywords", [])

    result = {
        "id": item["id"],
        "question": question,
        "topic": item.get("topic", ""),
        "difficulty": item.get("difficulty", ""),
    }

    t_start = time.monotonic()

    # 1. Rewrite (empty chat history for eval)
    rewritten = await simple_rag.arewrite_query(question, "История пуста.", "Новый пользователь.")

    # 2. Retrieve
    docs = await simple_rag.aretrieve(rewritten, top_k=top_k)

    # 3. Grade
    graded_docs = await simple_rag.agrade_documents(rewritten, docs)

    # 4. Compute retrieval recall
    all_doc_text = " ".join(d.page_content for d in docs)
    graded_doc_text = " ".join(d.page_content for d in graded_docs)

    result["retrieval_recall_raw"] = keyword_recall(expected_kw, all_doc_text)
    result["retrieval_recall_graded"] = keyword_recall(expected_kw, graded_doc_text)
    result["docs_retrieved"] = len(docs)
    result["docs_after_grading"] = len(graded_docs)

    # 5. Generate answer
    context_str = documents_to_context_str(graded_docs) if graded_docs else "Нет релевантных документов."
    
    answer_text = ""
    async for chunk in simple_rag.astream_answer(
        rewritten, context_str, "История пуста.", "Новый пользователь.", "10 марта 2026"
    ):
        answer_text += chunk

    # Parse JSON answer
    try:
        data = json.loads(answer_text)
        clean_answer = data.get("answer", answer_text)
        suggested = data.get("suggested_questions", [])
    except json.JSONDecodeError:
        clean_answer = answer_text
        suggested = []

    result["answer_keyword_coverage"] = keyword_recall(expected_kw, clean_answer)
    result["answer_length"] = len(clean_answer)
    result["suggested_count"] = len(suggested)

    t_end = time.monotonic()
    result["latency_s"] = round(t_end - t_start, 2)

    # 6. Simple hallucination check: does the answer contain claims not in context?
    # Heuristic: if the answer mentions specific URLs not in the context, flag it
    import re
    answer_urls = set(re.findall(r'https?://\S+', clean_answer))
    context_urls = set(re.findall(r'https?://\S+', context_str))
    hallucinated_urls = answer_urls - context_urls
    result["hallucinated_urls"] = list(hallucinated_urls)
    result["has_hallucinated_urls"] = len(hallucinated_urls) > 0

    return result


async def run_evaluation(dataset_path: str, top_k: int, output_path: str):
    """Run evaluation on the full dataset."""
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    logger.info("Loaded %d questions from %s", len(dataset), dataset_path)

    # Initialize RAG
    db_url, prompts = load_config()
    sync_url = db_url
    if "+asyncpg" in sync_url:
        sync_url = sync_url.replace("+asyncpg", "+psycopg")
    if sync_url.startswith("postgresql://"):
        sync_url = sync_url.replace("postgresql://", "postgresql+psycopg://", 1)

    engine = create_engine(sync_url)
    simple_rag = SimpleRAG(engine, prompts)

    logger.info("SimpleRAG initialized, starting evaluation...")

    results = []
    for i, item in enumerate(dataset):
        logger.info("[%d/%d] Evaluating: %s", i + 1, len(dataset), item["question"][:60])
        try:
            result = await evaluate_single(simple_rag, item, top_k=top_k)
            results.append(result)
            logger.info(
                "  → recall_raw=%.2f  recall_graded=%.2f  answer_kw=%.2f  latency=%.1fs",
                result["retrieval_recall_raw"],
                result["retrieval_recall_graded"],
                result["answer_keyword_coverage"],
                result["latency_s"],
            )
        except Exception as e:
            logger.error("  → FAILED: %s", e)
            results.append({
                "id": item["id"],
                "question": item["question"],
                "error": str(e),
            })

        # Rate limit: 2s between questions
        if i < len(dataset) - 1:
            await asyncio.sleep(2)

    # Compute aggregate metrics
    valid = [r for r in results if "error" not in r]
    if valid:
        avg_recall_raw = sum(r["retrieval_recall_raw"] for r in valid) / len(valid)
        avg_recall_graded = sum(r["retrieval_recall_graded"] for r in valid) / len(valid)
        avg_answer_kw = sum(r["answer_keyword_coverage"] for r in valid) / len(valid)
        avg_latency = sum(r["latency_s"] for r in valid) / len(valid)
        hallucination_rate = sum(1 for r in valid if r.get("has_hallucinated_urls")) / len(valid)

        summary = {
            "total_questions": len(dataset),
            "successful": len(valid),
            "failed": len(results) - len(valid),
            "avg_retrieval_recall_raw": round(avg_recall_raw, 3),
            "avg_retrieval_recall_graded": round(avg_recall_graded, 3),
            "avg_answer_keyword_coverage": round(avg_answer_kw, 3),
            "avg_latency_s": round(avg_latency, 2),
            "hallucination_rate": round(hallucination_rate, 3),
            "by_difficulty": {},
            "by_topic": {},
        }

        # Breakdowns
        for diff in ("basic", "medium", "hard"):
            subset = [r for r in valid if r.get("difficulty") == diff]
            if subset:
                summary["by_difficulty"][diff] = {
                    "count": len(subset),
                    "avg_recall": round(sum(r["retrieval_recall_graded"] for r in subset) / len(subset), 3),
                    "avg_answer_kw": round(sum(r["answer_keyword_coverage"] for r in subset) / len(subset), 3),
                }

        topics = set(r.get("topic", "") for r in valid)
        for topic in sorted(topics):
            subset = [r for r in valid if r.get("topic") == topic]
            if subset:
                summary["by_topic"][topic] = {
                    "count": len(subset),
                    "avg_recall": round(sum(r["retrieval_recall_graded"] for r in subset) / len(subset), 3),
                    "avg_answer_kw": round(sum(r["answer_keyword_coverage"] for r in subset) / len(subset), 3),
                }
    else:
        summary = {"error": "All questions failed"}

    report = {"summary": summary, "results": results}

    # Save report
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info("Questions: %d (%d OK, %d failed)", len(dataset), len(valid), len(results) - len(valid))
    if valid:
        logger.info("Avg Retrieval Recall (raw):     %.1f%%", avg_recall_raw * 100)
        logger.info("Avg Retrieval Recall (graded):  %.1f%%", avg_recall_graded * 100)
        logger.info("Avg Answer Keyword Coverage:    %.1f%%", avg_answer_kw * 100)
        logger.info("Avg Latency:                    %.1fs", avg_latency)
        logger.info("Hallucination Rate (URLs):      %.1f%%", hallucination_rate * 100)
    logger.info("Report saved to: %s", output_path)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality")
    parser.add_argument(
        "--dataset", default="tests/eval_dataset.json",
        help="Path to evaluation dataset JSON",
    )
    parser.add_argument(
        "--top-k", type=int, default=6,
        help="Number of documents to retrieve per query",
    )
    parser.add_argument(
        "--output", default="outputs/eval_report.json",
        help="Path to save the evaluation report",
    )
    args = parser.parse_args()

    asyncio.run(run_evaluation(args.dataset, args.top_k, args.output))


if __name__ == "__main__":
    main()
