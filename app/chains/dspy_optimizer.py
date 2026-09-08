import os
import json
from dotenv import load_dotenv
import dspy
from dspy.teleprompt import BootstrapFewShot

load_dotenv()

# ─────────────────────────────────────────
# 1. Configure DSPy to use Claude
# ─────────────────────────────────────────

def configure_dspy():
    """Configure DSPy to use Claude Haiku via Anthropic API."""
    lm = dspy.LM(
        model="anthropic/claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )
    dspy.configure(lm=lm)
    print("DSPy configured with Claude Haiku")


# ─────────────────────────────────────────
# 2. Define the DSPy signature
# ─────────────────────────────────────────

class FinancialQA(dspy.Signature):
    """
    Answer questions about financial documents
    using only the provided context.
    Cite specific figures, policies, and thresholds.
    If the context does not contain the answer, say so clearly.
    """
    context: str = dspy.InputField(
        desc="Relevant passages from financial documents"
    )
    question: str = dspy.InputField(
        desc="Analyst's question about the financial documents"
    )
    answer: str = dspy.OutputField(
        desc="Grounded answer citing specific figures from the context"
    )


# ─────────────────────────────────────────
# 3. Define the DSPy module
# ─────────────────────────────────────────

class FinancialRAGModule(dspy.Module):
    """RAG module that DSPy will optimise."""

    def __init__(self):
        self.generate = dspy.ChainOfThought(FinancialQA)

    def forward(self, context: str, question: str) -> dspy.Prediction:
        return self.generate(context=context, question=question)


# ─────────────────────────────────────────
# 4. Golden dataset
# ─────────────────────────────────────────

GOLDEN_DATASET = [
    {
        "context": """[Source: q3_2024_report.txt | Classification: PUBLIC]
WESTPAC Q3 2024 FINANCIAL REPORT
Net profit after tax: 890 million dollars, net margin 21%.
Revenue: 4.2 billion dollars, up 12% year-on-year.
Total loan book: 180 billion dollars.""",
        "question": "What is Westpac's net profit and revenue for Q3 2024?",
        "answer": "Westpac's net profit after tax for Q3 2024 is 890 million dollars with a net margin of 21%. Total revenue was 4.2 billion dollars, up 12% year-on-year."
    },
    {
        "context": """[Source: q3_2024_report.txt | Classification: PUBLIC]
Capital adequacy ratio CET1: 13.2%, above APRA minimum of 10.5%.
Liquidity coverage ratio: 138%, well above 100% regulatory minimum.
Net stable funding ratio: 121%.""",
        "question": "What is Westpac's capital adequacy ratio and how does it compare to the regulatory minimum?",
        "answer": "Westpac's CET1 capital adequacy ratio is 13.2%, which is 2.7 percentage points above the APRA minimum requirement of 10.5%."
    },
    {
        "context": """[Source: risk_framework.txt | Classification: INTERNAL]
Residential mortgage LVR limit: 80% without LMI.
All residential mortgage applications above 80 percent LVR require lenders mortgage insurance.
Maximum single counterparty exposure: 10% of capital base.""",
        "question": "What is the LVR limit for residential mortgages?",
        "answer": "The LVR limit for residential mortgages is 80%. Any application above 80% LVR requires lenders mortgage insurance (LMI)."
    },
    {
        "context": """[Source: risk_framework.txt | Classification: INTERNAL]
Cyber incidents must be reported within 72 hours to APRA.
Business continuity RTO: 4 hours for critical systems.
Third party vendor risk reviews: annual mandatory assessment.
Westpac experienced 847 security incidents in the 12 months to September 2024.""",
        "question": "What are the cyber security reporting requirements?",
        "answer": "Cyber incidents must be reported to APRA within 72 hours. Westpac experienced 847 security incidents in the 12 months to September 2024, of which 12 were severity 1 requiring immediate board escalation."
    },
    {
        "context": """[Source: digital_strategy.txt | Classification: CONFIDENTIAL]
Total investment: 1.5 billion dollars over three financial years.
FY2024 investment: 420 million dollars.
FY2025 investment: 580 million dollars.
FY2026 investment: 500 million dollars.""",
        "question": "How much is Westpac investing in digital transformation?",
        "answer": "Westpac is investing 1.5 billion dollars total over three financial years: 420 million in FY2024, 580 million in FY2025, and 500 million in FY2026."
    },
    {
        "context": """[Source: digital_strategy.txt | Classification: CONFIDENTIAL]
Mortgage Processing AI:
The mortgage processing AI system represents the largest single AI investment at 95 million dollars.
Expected to reduce processing time from 5.2 business days to 4 hours for standard applications.
Human review required for applications above 2 million dollars or confidence score below 85 percent.""",
        "question": "How much is allocated to mortgage processing AI and what are the expected benefits?",
        "answer": "The mortgage processing AI system receives 95 million dollars, the largest single AI investment. It is expected to reduce approval time from 5.2 business days to 4 hours for standard applications. Human review is required for applications above 2 million dollars or where confidence falls below 85%."
    },
    {
        "context": """[Source: economic_outlook.txt | Classification: PUBLIC]
RBA cash rate: expected to remain at 4.35% through Q1 2025.
First rate cut forecast: May 2025, 25bps reduction.
Full easing cycle expected to deliver 100 basis points of cuts by end of 2025.""",
        "question": "When does Westpac Economics forecast the first RBA rate cut?",
        "answer": "Westpac Economics forecasts the first RBA rate cut of 25 basis points in May 2025, with the full easing cycle delivering 100 basis points of cuts by end of 2025."
    },
    {
        "context": """[Source: economic_outlook.txt | Classification: PUBLIC]
Sydney median house price: 1.47 million dollars, up 6% year-on-year.
Melbourne median: 920,000 dollars, up 2% year-on-year.
Brisbane median: 870,000 dollars, up 14% year-on-year.""",
        "question": "What are the current median house prices in Sydney, Melbourne, and Brisbane?",
        "answer": "Sydney median house price is 1.47 million dollars (up 6% year-on-year), Melbourne is 920,000 dollars (up 2%), and Brisbane is 870,000 dollars (up 14%)."
    },
    {
        "context": """[Source: q3_2024_report.txt | Classification: PUBLIC]
90-day mortgage arrears increased to 0.89 percent of the mortgage portfolio from 0.72 percent in the prior quarter.
Hardship registrations increased 23 percent quarter on quarter to 18,400 active cases.
78 percent of customers who complete the hardship program return to regular repayment schedule.""",
        "question": "What is the current mortgage arrears rate?",
        "answer": "90-day mortgage arrears have risen to 0.89% from 0.72% in the prior quarter. Hardship registrations increased 23% to 18,400 active cases, with 78% of customers completing the hardship program returning to regular repayments."
    },
    {
        "context": """[Source: risk_framework.txt | Classification: INTERNAL]
VaR limit: 50 million dollars at 99% confidence interval.
Interest rate sensitivity: 100bps parallel shift impact 380 million dollars.
Foreign exchange exposure limit: 2 billion dollars net open position.""",
        "question": "What are the market risk limits?",
        "answer": "The market risk limits are: VaR limit of 50 million dollars at 99% confidence interval, interest rate sensitivity of 380 million dollars for a 100bps parallel shift, and foreign exchange net open position limit of 2 billion dollars."
    },
]


# ─────────────────────────────────────────
# 5. Faithfulness metric
# ─────────────────────────────────────────

def faithfulness_metric(example, prediction, trace=None) -> float:
    """
    Measure how faithful the answer is to the context.
    Simple keyword overlap metric — RAGAS integration in eval suite.
    """
    answer = prediction.answer.lower()
    expected = example.answer.lower()
    context = example.context.lower()

    # Check key numbers and terms from expected answer appear in prediction
    expected_words = set(expected.split())
    answer_words = set(answer.split())

    # Numbers are especially important for financial QA
    numbers_in_expected = {w for w in expected_words if any(c.isdigit() for c in w)}
    numbers_in_answer = {w for w in answer_words if any(c.isdigit() for c in w)}

    if not numbers_in_expected:
        # No numbers — use word overlap
        overlap = len(expected_words & answer_words) / len(expected_words)
        return overlap > 0.3

    # Check numeric facts are preserved
    number_overlap = len(numbers_in_expected & numbers_in_answer)
    number_score = number_overlap / len(numbers_in_expected)

    return number_score >= 0.5


# ─────────────────────────────────────────
# 6. Build training examples
# ─────────────────────────────────────────

def build_training_examples():
    """Convert golden dataset to DSPy examples."""
    examples = []
    for item in GOLDEN_DATASET:
        example = dspy.Example(
            context=item["context"],
            question=item["question"],
            answer=item["answer"],
        ).with_inputs("context", "question")
        examples.append(example)
    return examples


# ─────────────────────────────────────────
# 7. Run optimisation
# ─────────────────────────────────────────

def optimise_prompt(save_path: str = "storage/optimised_prompt.json"):
    """
    Run DSPy BootstrapFewShot optimisation.
    Finds the best prompt for financial QA
    based on the faithfulness metric.
    """
    configure_dspy()

    print("\nBuilding training examples...")
    examples = build_training_examples()
    print(f"Training examples: {len(examples)}")

    # Split into train and validation
    train = examples[:7]
    val = examples[7:]

    print("\nInitialising FinancialRAGModule...")
    module = FinancialRAGModule()

    print("\nRunning BootstrapFewShot optimisation...")
    print("This will make several LLM calls — expect ~$0.10-0.30")

    optimiser = BootstrapFewShot(
        metric=faithfulness_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
    )

    compiled_module = optimiser.compile(
        module,
        trainset=train,
    )

    print("\nOptimisation complete")

    # Test on validation set
    print("\nValidation results:")
    scores = []
    for example in val:
        prediction = compiled_module(
            context=example.context,
            question=example.question,
        )
        score = faithfulness_metric(example, prediction)
        scores.append(score)
        print(f"Q: {example.question[:60]}...")
        print(f"A: {prediction.answer[:100]}...")
        print(f"Score: {score}")
        print()

    avg_score = sum(scores) / len(scores)
    print(f"Average validation score: {avg_score:.2f}")

    # Save the optimised module
    os.makedirs("storage", exist_ok=True)
    compiled_module.save(save_path)
    print(f"Optimised prompt saved to {save_path}")

    return compiled_module, avg_score


def load_optimised_module(path: str = "storage/optimised_prompt.json"):
    """Load a previously optimised DSPy module."""
    configure_dspy()
    module = FinancialRAGModule()
    module.load(path)
    print(f"Optimised module loaded from {path}")
    return module


def compare_prompts(retriever, test_question: str):
    """
    Compare baseline vs optimised prompt on a test question.
    Shows the improvement DSPy achieves.
    """
    configure_dspy()

    # Get context from retriever
    nodes = retriever.retrieve(test_question)
    context = "\n\n".join([
        f"[Source: {n.metadata.get('file_name')} | "
        f"Classification: {n.metadata.get('classification')}]\n{n.text}"
        for n in nodes
    ])

    print("=" * 50)
    print("BASELINE vs OPTIMISED COMPARISON")
    print("=" * 50)

    # Baseline
    baseline = FinancialRAGModule()
    baseline_result = baseline(context=context, question=test_question)
    print(f"\nQuestion: {test_question}")
    print(f"\nBaseline answer:\n{baseline_result.answer}")

    # Optimised
    try:
        optimised = load_optimised_module()
        optimised_result = optimised(context=context, question=test_question)
        print(f"\nOptimised answer:\n{optimised_result.answer}")
    except Exception:
        print("\nNo optimised module found — run optimise_prompt() first")


if __name__ == "__main__":
    compiled_module, score = optimise_prompt()
    print(f"\nFinal average score: {score:.2f}")