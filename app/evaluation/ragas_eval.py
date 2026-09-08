import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from app.indexing.indexer import load_index, get_query_engine, ROLE_PERMISSIONS

load_dotenv()

Settings.llm = Anthropic(
    model="claude-3-5-haiku-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

# ─────────────────────────────────────────
# Golden evaluation dataset
# ─────────────────────────────────────────

EVAL_DATASET = [
    {
        "question": "What is Westpac's net profit after tax for Q3 2024?",
        "ground_truth": "Westpac's net profit after tax for Q3 2024 is 890 million dollars with a net margin of 21%.",
        "role": "public",
    },
    {
        "question": "What is Westpac's capital adequacy CET1 ratio?",
        "ground_truth": "Westpac's CET1 capital adequacy ratio is 13.2%, above the APRA minimum of 10.5%.",
        "role": "public",
    },
    {
        "question": "What is the 90-day mortgage arrears rate?",
        "ground_truth": "90-day mortgage arrears increased to 0.89% from 0.72% in the prior quarter.",
        "role": "public",
    },
    {
        "question": "What is the RBA cash rate forecast?",
        "ground_truth": "The RBA cash rate is expected to remain at 4.35% through Q1 2025 with first cut forecast in May 2025.",
        "role": "public",
    },
    {
        "question": "What is the LVR limit for residential mortgages?",
        "ground_truth": "The LVR limit is 80%. Applications above 80% LVR require lenders mortgage insurance.",
        "role": "junior_analyst",
    },
    {
        "question": "What are the cyber security reporting requirements to APRA?",
        "ground_truth": "Cyber incidents must be reported to APRA within 72 hours.",
        "role": "junior_analyst",
    },
    {
        "question": "What is the VaR limit for market risk?",
        "ground_truth": "The VaR limit is 50 million dollars at 99% confidence interval.",
        "role": "junior_analyst",
    },
    {
        "question": "What is the total digital transformation investment?",
        "ground_truth": "Westpac is investing 1.5 billion dollars over three financial years in digital transformation.",
        "role": "senior_analyst",
    },
    {
        "question": "How much is allocated to mortgage processing AI?",
        "ground_truth": "The mortgage processing AI system receives 95 million dollars, the largest single AI investment.",
        "role": "senior_analyst",
    },
    {
        "question": "What is the expected payback period for the digital transformation program?",
        "ground_truth": "The payback period is 2.6 years from program completion.",
        "role": "senior_analyst",
    },
]


# ─────────────────────────────────────────
# Score extraction helper
# ─────────────────────────────────────────

def safe_score(val):
    """Handle both float and list returns from RAGAS."""
    if isinstance(val, list):
        valid = [v for v in val if v is not None]
        return round(sum(valid) / len(valid), 3) if valid else 0.0
    return round(float(val), 3)


# ─────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────

def run_evaluation(save_results: bool = True):
    """
    Run RAGAS evaluation on the golden dataset.
    Measures faithfulness, answer relevancy,
    context precision, and context recall.
    Uses Claude as the judge LLM instead of OpenAI.
    """
    print("\nLoading index from Neon...")
    index = load_index()

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    latencies = []

    print(f"\nRunning evaluation on {len(EVAL_DATASET)} questions...")
    print("=" * 50)

    for i, item in enumerate(EVAL_DATASET):
        question = item["question"]
        role = item["role"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i+1}/{len(EVAL_DATASET)}] Role: {role}")
        print(f"Q: {question}")

        allowed = ROLE_PERMISSIONS[role]
        filters = MetadataFilters(filters=[
            MetadataFilter(
                key="classification",
                value=allowed,
                operator=FilterOperator.IN,
            )
        ])

        retriever = index.as_retriever(
            similarity_top_k=3,
            filters=filters,
        )

        start = time.time()
        engine = get_query_engine(index, role)
        answer = str(engine.query(question))
        latency = time.time() - start

        nodes = retriever.retrieve(question)
        context = [n.text for n in nodes]

        print(f"A: {answer[:100]}...")
        print(f"Latency: {latency:.2f}s")

        questions.append(question)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)
        latencies.append(latency)

    print("\n" + "=" * 50)
    print("Running RAGAS metrics using Claude as judge...")
    print("=" * 50)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Configure RAGAS to use Claude instead of OpenAI
    ragas_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1024,
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    )

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(EVAL_DATASET),
        "metrics": {
            "faithfulness": safe_score(results["faithfulness"]),
            "answer_relevancy": safe_score(results["answer_relevancy"]),
            "context_precision": safe_score(results["context_precision"]),
            "context_recall": safe_score(results["context_recall"]),
        },
        "performance": {
            "avg_latency_s": round(sum(latencies) / len(latencies), 2),
            "max_latency_s": round(max(latencies), 2),
            "min_latency_s": round(min(latencies), 2),
        },
    }

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    print(f"Faithfulness:       {summary['metrics']['faithfulness']:.3f}")
    print(f"Answer Relevancy:   {summary['metrics']['answer_relevancy']:.3f}")
    print(f"Context Precision:  {summary['metrics']['context_precision']:.3f}")
    print(f"Context Recall:     {summary['metrics']['context_recall']:.3f}")
    print(f"\nAvg Latency:        {summary['performance']['avg_latency_s']:.2f}s")
    print(f"Max Latency:        {summary['performance']['max_latency_s']:.2f}s")
    print(f"Min Latency:        {summary['performance']['min_latency_s']:.2f}s")

    if save_results:
        os.makedirs("storage", exist_ok=True)
        path = "storage/eval_results.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {path}")

    return summary


# ─────────────────────────────────────────
# Quality gate for CI/CD
# ─────────────────────────────────────────

def check_quality_gate(results_path: str = "storage/eval_results.json"):
    """
    CI/CD quality gate.
    Returns True if all metrics pass thresholds.
    Used by GitHub Actions to block bad deployments.
    """
    THRESHOLDS = {
        "faithfulness": 0.7,
        "answer_relevancy": 0.7,
        "context_precision": 0.5,
        "context_recall": 0.5,
    }

    with open(results_path) as f:
        results = json.load(f)

    metrics = results["metrics"]
    passed = True
    failed_metrics = []

    print("\nQUALITY GATE CHECK")
    print("=" * 40)

    for metric, threshold in THRESHOLDS.items():
        value = metrics.get(metric, 0)
        status = "✅ PASS" if value >= threshold else "❌ FAIL"
        if value < threshold:
            passed = False
            failed_metrics.append(metric)
        print(f"{metric}: {value:.3f} "
              f"(threshold: {threshold}) {status}")

    print("=" * 40)
    if passed:
        print("✅ All metrics passed — deployment approved")
    else:
        print(f"❌ Failed metrics: {failed_metrics}")
        print("🚫 Deployment blocked")

    return passed


if __name__ == "__main__":
    summary = run_evaluation()
    passed = check_quality_gate()
    exit(0 if passed else 1)