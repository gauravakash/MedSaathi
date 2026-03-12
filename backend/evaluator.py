"""
MedSaathi - Offline RAG Evaluation with RAGAS
=============================================

Why this file exists:
  RAG improvement without evaluation is guesswork.
  - "Did reranking actually help?" Without metrics, no proof.
  - "Is chunking size good?" Without metrics, no proof.
  - "Did prompt changes reduce hallucinations?" Without metrics, no proof.

This module runs MedSaathi's full RAG pipeline on a fixed test set and
produces measurable scores using RAGAS. These scores become a quality baseline
that we can compare across pipeline versions.

RAGAS metrics used here (0 to 1 scale):
  1) Faithfulness:
     Does the answer stay grounded in retrieved context?
     Medical example: if context does NOT contain a dosage, the answer should
     not invent one. Low faithfulness means hallucination risk.

  2) Answer Relevancy:
     Does the answer actually answer the user's question?
     Medical example: user asks ORS steps, answer should not drift into general
     stomach infection advice only.

  3) Context Recall:
     Did retrieval include the information needed to answer correctly?
     Medical example: TB duration asked, but retrieval misses duration chunk.

  4) Context Precision:
     How much retrieved context was actually useful vs noise?
     Medical example: top chunks should focus on BP control, not unrelated topics.

LLM-as-judge pattern:
  RAGAS uses an LLM to judge quality (here: Gemini Flash). It effectively asks
  questions like, "Is this answer faithful to the supplied context?" and scores
  responses systematically.

Offline vs online evaluation:
  - Offline (this file): repeatable benchmark on a fixed golden dataset.
  - Online (production): real-user feedback, click-through, correction rates,
    escalation rate, and safety incident tracking.

Operational note:
  Run evaluator.py after EVERY major pipeline change: chunk size, embedding
  model, query rewriting prompt, reranker threshold/top-k, or generator prompt.
"""

from __future__ import annotations

import json
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# RAGAS imports are optional at import-time so this module can still be imported
# in environments where ragas is not installed yet.
try:
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR: Exception | None = None
except Exception as import_error:  # pragma: no cover - runtime environment dependent
    evaluate = None
    LangchainLLMWrapper = None
    faithfulness = None
    answer_relevancy = None
    context_recall = None
    context_precision = None
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = import_error

# Make absolute backend.* imports work when this file is run directly:
#   uv run backend/evaluator.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.generator import GENERATOR_MODEL, generate, get_generator_llm
from backend.query_rewriter import process_query
from backend.reranker import RERANKER_MODEL, rerank
from backend.retriever import hybrid_search

# Optional: import ingestion constants for result metadata.
try:
    from backend.ingest import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
except Exception:  # pragma: no cover - defensive fallback
    CHUNK_SIZE = None
    CHUNK_OVERLAP = None
    EMBEDDING_MODEL = "unknown"


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

EVAL_RESULTS_DIR = "./eval_results"
DEFAULT_EVAL_MODEL = "gemini-1.5-flash"
SCORE_THRESHOLDS = {
    "good": 0.8,
    "acceptable": 0.6,
    "poor": 0.0,
}

# Keep aligned with main pipeline defaults.
RETRIEVAL_N_RESULTS = 10
RERANK_TOP_N = 5

console = Console(force_terminal=True)


# ----------------------------------------------------------------------------
# Golden evaluation dataset
# ----------------------------------------------------------------------------

# This is the project's "golden dataset" for offline evaluation.
# IMPORTANT: before production use, ground_truth answers should be reviewed and
# verified by a qualified medical professional.
EVAL_DATASET: list[dict[str, str]] = [
    {
        "question": "Bukhar mein paracetamol 500mg kitni baar le sakte hain?",
        "ground_truth": (
            "Adults usually take 500mg to 1000mg every 4 to 6 hours as needed, "
            "and should not exceed 4000mg in 24 hours. People with liver disease, "
            "alcohol use, pregnancy, or child dosing needs should consult a doctor."
        ),
    },
    {
        "question": "What should a diabetic patient eat daily?",
        "ground_truth": (
            "A diabetes-friendly diet emphasizes whole grains, pulses, vegetables, "
            "lean protein, controlled portion sizes, and reduced sugary drinks and "
            "refined carbs. Regular meal timing and clinician-guided planning are important."
        ),
    },
    {
        "question": "Bacche ka vaccination schedule pehle saal mein kya hota hai?",
        "ground_truth": (
            "In the first year, routine vaccines are given at birth and then at "
            "6, 10, and 14 weeks, with additional doses around 9 months as per "
            "national schedule. Parents should follow the official immunization card and PHC advice."
        ),
    },
    {
        "question": "Loose motion mein ORS kaise dena chahiye?",
        "ground_truth": (
            "Prepare ORS exactly as instructed on the packet with clean water. "
            "Give small frequent sips after each loose stool, continue feeding, and seek urgent care "
            "if there is blood in stool, persistent vomiting, lethargy, or signs of dehydration."
        ),
    },
    {
        "question": "Malaria ke common symptoms kya hote hain?",
        "ground_truth": (
            "Common symptoms include fever with chills, sweating, headache, body ache, "
            "fatigue, nausea, and sometimes vomiting. Confirm diagnosis with a blood test "
            "and seek treatment early."
        ),
    },
    {
        "question": "How can high blood pressure be managed at home?",
        "ground_truth": (
            "Home management includes regular BP monitoring, reduced salt intake, "
            "daily physical activity, weight control, avoiding tobacco, limiting alcohol, "
            "stress management, and taking prescribed medicines consistently."
        ),
    },
    {
        "question": "Pregnancy mein nutrition ke liye kya khana chahiye?",
        "ground_truth": (
            "Pregnancy diet should include balanced meals with protein, iron, calcium, "
            "folate-rich foods, fruits, vegetables, and adequate hydration. Antenatal supplements "
            "and individualized doctor guidance are essential."
        ),
    },
    {
        "question": "TB ka treatment kitne mahine chalta hai?",
        "ground_truth": (
            "Drug-susceptible TB treatment usually lasts about 6 months under supervised protocols, "
            "but duration can vary by type of TB and patient response. Treatment should never be stopped early."
        ),
    },
]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _ensure_ragas_available() -> None:
    """Raise a clear error if RAGAS is unavailable in the current environment."""
    if not RAGAS_AVAILABLE:
        raise RuntimeError(
            "RAGAS is not installed or failed to import. "
            "Install dependencies first (for example: `uv sync`) and retry. "
            f"Import error: {RAGAS_IMPORT_ERROR}"
        )


def _clamp_score(score: float) -> float:
    """Clamp numeric score into [0, 1] for display stability."""
    return max(0.0, min(1.0, float(score)))


def _score_color(score: float) -> str:
    """Map a score to display color based on configured thresholds."""
    score = _clamp_score(score)
    if score >= SCORE_THRESHOLDS["good"]:
        return "green"
    if score >= SCORE_THRESHOLDS["acceptable"]:
        return "yellow"
    return "red"


def _ascii_bar(score: float, width: int = 10) -> str:
    """Build a simple ASCII progress bar for a 0-1 metric."""
    score = _clamp_score(score)
    filled = int(round(score * width))
    return "#" * filled + "-" * (width - filled)


def _extract_metric_scores(ragas_result: Any) -> dict[str, float]:
    """
    Extract aggregate metric scores from different possible RAGAS result shapes.

    RAGAS versions expose results differently (mapping, pandas table, score dict,
    or per-row records). This helper normalizes into a stable metric dict.
    """
    metric_keys = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]

    scores: dict[str, float] = {key: 0.0 for key in metric_keys}
    found: dict[str, bool] = {key: False for key in metric_keys}

    def capture(metric_name: str, value: Any) -> None:
        try:
            if value is None:
                return
            scores[metric_name] = float(value)
            found[metric_name] = True
        except (TypeError, ValueError):
            return

    # 1) Mapping-like result
    if isinstance(ragas_result, dict):
        for key in metric_keys:
            capture(key, ragas_result.get(key))

    # 2) Key access result
    for key in metric_keys:
        try:
            capture(key, ragas_result[key])
        except Exception:
            pass

    # 3) to_pandas() result with metric columns
    if hasattr(ragas_result, "to_pandas"):
        try:
            df = ragas_result.to_pandas()
            for key in metric_keys:
                if key in df.columns:
                    non_null_values = [
                        float(val) for val in df[key].tolist() if val is not None
                    ]
                    if non_null_values:
                        capture(key, sum(non_null_values) / len(non_null_values))
        except Exception:
            pass

    # 4) .scores result (dict or list of records)
    raw_scores = getattr(ragas_result, "scores", None)
    if isinstance(raw_scores, dict):
        for key in metric_keys:
            capture(key, raw_scores.get(key))
    elif isinstance(raw_scores, list):
        for key in metric_keys:
            values = []
            for row in raw_scores:
                if isinstance(row, dict) and row.get(key) is not None:
                    try:
                        values.append(float(row[key]))
                    except (TypeError, ValueError):
                        continue
            if values:
                capture(key, sum(values) / len(values))

    # Warn if any metric is missing after extraction attempts.
    missing = [key for key, is_found in found.items() if not is_found]
    if missing:
        console.print(
            f"[yellow]Warning:[/yellow] Could not confidently extract metrics: {missing}. "
            "They were defaulted to 0.0"
        )

    return {key: _clamp_score(value) for key, value in scores.items()}


def _run_pipeline(question: str) -> dict[str, Any]:
    """
    Run the full MedSaathi RAG pipeline for one question.

    Steps:
      1) query rewrite
      2) hybrid retrieval
      3) reranking
      4) grounded generation
    """
    try:
        query_info = process_query(question)
        rewritten_query = query_info["best"]
        language = query_info["language"]

        retrieved_chunks = hybrid_search(
            rewritten_query,
            n_results=RETRIEVAL_N_RESULTS,
        )
        reranked_chunks = rerank(
            rewritten_query,
            retrieved_chunks,
            top_n=RERANK_TOP_N,
        )

        generation = generate(
            query=question,
            chunks=reranked_chunks,
            language=language,
            history=[],
        )

        contexts = [chunk.get("text", "") for chunk in reranked_chunks if chunk.get("text")]
        top_chunk_preview = ""
        if contexts:
            top_chunk_preview = contexts[0][:200]
            if len(contexts[0]) > 200:
                top_chunk_preview += "..."

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": generation.get("answer", ""),
            "contexts": contexts,
            "chunks_retrieved": len(reranked_chunks),
            "top_chunk_preview": top_chunk_preview,
        }
    except Exception as exc:  # pragma: no cover - runtime pipeline dependent
        console.print(
            f"[red]Pipeline failed[/red] for question '{question[:60]}...': "
            f"{type(exc).__name__}: {exc}"
        )
        return {
            "question": question,
            "rewritten_query": question,
            "answer": "",
            "contexts": [],
            "chunks_retrieved": 0,
            "top_chunk_preview": "",
        }


# ----------------------------------------------------------------------------
# Core API
# ----------------------------------------------------------------------------


def build_ragas_dataset(
    questions: list[str],
    ground_truths: list[str],
) -> Dataset:
    """
    Build a HuggingFace Dataset with full-pipeline outputs for RAGAS.

    Why all 4 columns are required:
      - question: what the user asked
      - answer: what your system generated
      - contexts: what your retriever gave to generation
      - ground_truth: what the answer should ideally contain

    RAGAS combines these to score faithfulness, relevancy, recall, and precision.
    """
    if len(questions) != len(ground_truths):
        raise ValueError(
            "questions and ground_truths must have the same length, "
            f"got {len(questions)} and {len(ground_truths)}"
        )

    built_questions: list[str] = []
    built_answers: list[str] = []
    built_contexts: list[list[str]] = []
    built_ground_truths: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Building RAGAS dataset", total=len(questions))

        for question, ground_truth in zip(questions, ground_truths):
            sample = _run_pipeline(question)

            built_questions.append(question)
            built_answers.append(sample["answer"])
            built_contexts.append(sample["contexts"])
            built_ground_truths.append(ground_truth)

            progress.advance(task_id)

    return Dataset.from_dict(
        {
            "question": built_questions,
            "answer": built_answers,
            "contexts": built_contexts,
            "ground_truth": built_ground_truths,
        }
    )


def run_evaluation(dataset: Dataset | None = None) -> dict[str, Any]:
    """
    Run RAGAS evaluation and return aggregated scores.

    RAGAS uses an LLM judge to evaluate qualitative dimensions. Here we wrap
    Gemini Flash (via LangChain) and let RAGAS ask judge prompts like:
    "Does this answer faithfully reflect the given context?"
    """
    _ensure_ragas_available()

    if dataset is None:
        questions = [row["question"] for row in EVAL_DATASET]
        ground_truths = [row["ground_truth"] for row in EVAL_DATASET]
        dataset = build_ragas_dataset(questions, ground_truths)

    llm_wrapper = LangchainLLMWrapper(get_generator_llm())
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    console.print("\n[bold cyan]Running RAGAS evaluation...[/bold cyan]")

    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm_wrapper,
        )
    except TypeError:
        # Compatibility fallback for alternate evaluate signatures.
        ragas_result = evaluate(dataset, metrics=metrics, llm=llm_wrapper)

    metric_scores = _extract_metric_scores(ragas_result)

    overall_score = sum(metric_scores.values()) / len(metric_scores)

    return {
        "faithfulness": metric_scores["faithfulness"],
        "answer_relevancy": metric_scores["answer_relevancy"],
        "context_recall": metric_scores["context_recall"],
        "context_precision": metric_scores["context_precision"],
        "overall_score": _clamp_score(overall_score),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": len(dataset),
        "judge_model": DEFAULT_EVAL_MODEL,
    }


def print_evaluation_report(results: dict[str, Any]) -> None:
    """Print a color-coded evaluation report with ASCII bars and interpretations."""
    faith = _clamp_score(float(results.get("faithfulness", 0.0)))
    relevancy = _clamp_score(float(results.get("answer_relevancy", 0.0)))
    recall = _clamp_score(float(results.get("context_recall", 0.0)))
    precision = _clamp_score(float(results.get("context_precision", 0.0)))
    overall = _clamp_score(float(results.get("overall_score", (faith + relevancy + recall + precision) / 4.0)))

    rows = [
        ("Faithfulness", faith),
        ("Answer Relevancy", relevancy),
        ("Context Recall", recall),
        ("Context Precision", precision),
    ]

    console.print("\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║     MedSaathi RAG Evaluation Report      ║[/bold cyan]")
    console.print("[bold cyan]╠══════════════════════════════════════════╣[/bold cyan]")

    for label, score in rows:
        color = _score_color(score)
        bar = _ascii_bar(score)
        console.print(
            f"[bold cyan]║[/bold cyan]  "
            f"{label:<17} [{color}]{score:0.2f}[/{color}]  [{color}]{bar}[/{color}]  "
            f"[bold cyan]║[/bold cyan]"
        )

    overall_color = _score_color(overall)
    overall_bar = _ascii_bar(overall)

    console.print("[bold cyan]╠══════════════════════════════════════════╣[/bold cyan]")
    console.print(
        f"[bold cyan]║[/bold cyan]  Overall Score     "
        f"[{overall_color}]{overall:0.2f}[/{overall_color}]  "
        f"[{overall_color}]{overall_bar}[/{overall_color}]  "
        f"[bold cyan]║[/bold cyan]"
    )
    console.print("[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]")

    console.print("\n[bold]Interpretation:[/bold]")
    any_warning = False

    if faith < 0.8:
        console.print("[yellow]⚠️  Hallucination risk - review chunking[/yellow]")
        any_warning = True

    if recall < 0.7:
        console.print("[yellow]⚠️  Missing info - add more documents[/yellow]")
        any_warning = True

    if precision < 0.7:
        console.print("[yellow]⚠️  Noisy retrieval - tune chunk size[/yellow]")
        any_warning = True

    if relevancy < 0.8:
        console.print("[yellow]⚠️  Off-topic answers - review prompt[/yellow]")
        any_warning = True

    if not any_warning:
        console.print("[green]All core metrics are in a healthy range.[/green]")


def save_results(results: dict[str, Any], filepath: str | None = None) -> str:
    """
    Save evaluation output to JSON.

    Save every run so you can compare quality trends over time while tuning
    the pipeline.
    """
    timestamp_for_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if filepath is None:
        filepath = str(Path(EVAL_RESULTS_DIR) / f"eval_{timestamp_for_name}.json")

    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": int(results.get("num_questions", len(EVAL_DATASET))),
        "metrics": {
            "faithfulness": _clamp_score(float(results.get("faithfulness", 0.0))),
            "answer_relevancy": _clamp_score(float(results.get("answer_relevancy", 0.0))),
            "context_recall": _clamp_score(float(results.get("context_recall", 0.0))),
            "context_precision": _clamp_score(float(results.get("context_precision", 0.0))),
            "overall_score": _clamp_score(float(results.get("overall_score", 0.0))),
        },
        "pipeline_config": {
            "judge_model": DEFAULT_EVAL_MODEL,
            "generator_model": GENERATOR_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "retrieval_n_results": RETRIEVAL_N_RESULTS,
            "rerank_top_n": RERANK_TOP_N,
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    return str(output_path)


def _load_run(filepath: str) -> dict[str, Any]:
    """Load one saved evaluation run JSON."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_metrics_from_saved_run(saved: dict[str, Any]) -> dict[str, float]:
    """Normalize metrics section from saved JSON into a plain metric map."""
    if "metrics" in saved and isinstance(saved["metrics"], dict):
        source = saved["metrics"]
    else:
        source = saved

    return {
        "faithfulness": _clamp_score(float(source.get("faithfulness", 0.0))),
        "answer_relevancy": _clamp_score(float(source.get("answer_relevancy", 0.0))),
        "context_recall": _clamp_score(float(source.get("context_recall", 0.0))),
        "context_precision": _clamp_score(float(source.get("context_precision", 0.0))),
    }


def compare_runs(filepath1: str, filepath2: str) -> None:
    """Compare two saved evaluation runs and print metric deltas."""
    run1 = _load_run(filepath1)
    run2 = _load_run(filepath2)

    metrics1 = _extract_metrics_from_saved_run(run1)
    metrics2 = _extract_metrics_from_saved_run(run2)

    table = Table(title="MedSaathi Evaluation Run Comparison", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Run 1", justify="right")
    table.add_column("Run 2", justify="right")
    table.add_column("Change", justify="right")

    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    for key, label in metric_labels.items():
        score1 = metrics1[key]
        score2 = metrics2[key]
        delta = score2 - score1

        if delta > 0.001:
            style = "green"
            marker = "✅"
        elif delta < -0.001:
            style = "red"
            marker = "⚠️"
        else:
            style = "yellow"
            marker = "~"

        table.add_row(
            label,
            f"{score1:.2f}",
            f"{score2:.2f}",
            f"[{style}]{delta:+.2f} {marker}[/{style}]",
        )

    console.print(table)


def evaluate_single(question: str, ground_truth: str) -> dict[str, Any]:
    """
    Evaluate one question for debugging failing cases quickly.

    Returns both pipeline internals (rewrite/chunk preview) and per-question
    RAGAS metrics (faithfulness + answer relevancy).
    """
    _ensure_ragas_available()

    sample = _run_pipeline(question)

    single_dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [sample["answer"]],
            "contexts": [sample["contexts"]],
            "ground_truth": [ground_truth],
        }
    )

    llm_wrapper = LangchainLLMWrapper(get_generator_llm())

    try:
        single_result = evaluate(
            dataset=single_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm_wrapper,
        )
    except TypeError:
        single_result = evaluate(
            single_dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm_wrapper,
        )

    scores = _extract_metric_scores(single_result)

    return {
        "question": question,
        "rewritten_query": sample["rewritten_query"],
        "answer": sample["answer"],
        "chunks_retrieved": sample["chunks_retrieved"],
        "top_chunk_preview": sample["top_chunk_preview"],
        "faithfulness": scores.get("faithfulness", 0.0),
        "answer_relevancy": scores.get("answer_relevancy", 0.0),
    }


# ----------------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()

    console.print("[bold magenta]=== MedSaathi RAG Evaluator ===[/bold magenta]")

    try:
        evaluation_results = run_evaluation()
        print_evaluation_report(evaluation_results)

        saved_path = save_results(evaluation_results)

        elapsed_seconds = int(time.time() - start_time)
        minutes, seconds = divmod(elapsed_seconds, 60)

        console.print(
            f"\n[bold green]Evaluation complete in {minutes}m {seconds}s - "
            f"results saved to {saved_path}[/bold green]"
        )
    except Exception as exc:  # pragma: no cover - runtime error path
        console.print(f"[red]Evaluation failed:[/red] {type(exc).__name__}: {exc}")
        raise


