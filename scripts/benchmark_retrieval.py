from pathlib import Path

import pandas as pd

from src.chains.llm import get_llm
from src.chains.prompts import get_rag_prompt
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.dataset import FinanceBenchDataset
from src.evaluation.pipeline import RetrievalPipeline
from src.evaluation.retrieval import RetrievalEvaluator
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.retrieval.retrievers import create_retriever
from src.evaluation.splits import Split
from scripts.error_analysis import save_error_analysis
from src.evaluation.benchmark import BenchmarkWriter


VECTORSTORE_PATH = Path("./data/vectorstore")

DATASET_PATH = Path(
    "./data/financebench/data/financebench_open_source.jsonl"
)

embedding_model = get_embedding_model()

vectorstore = FAISSStore()

vectorstore.load(
    VECTORSTORE_PATH,
    embedding_model,
)

retriever = create_retriever(
    vectorstore.store,
)

pipeline = RetrievalPipeline(retriever)

runner = BenchmarkRunner(
    rag_pipeline=pipeline,
    retrieval_evaluator=RetrievalEvaluator(),
)

dataset = FinanceBenchDataset.from_jsonl(DATASET_PATH)
split = Split.load("data/financebench/splits")
dataset = dataset.subset(split.train_ids)

with BenchmarkWriter("results/retrieval.jsonl") as writer:

    for result in runner.run(dataset):
        writer.write(result)
results = runner.evaluate(dataset)
for benchmark_result in results[:5]:

    print("=" * 80)

    print("Question:")
    print(benchmark_result.sample.question)

    print()

    print("Expected:")
    print(benchmark_result.sample.source_document)

    print()

    print("Retrieved:")

    for doc in benchmark_result.result.retrieved_documents:

        print(
            f"{doc.rank:>2}. "
            f"{doc.document_id}"
        )

metrics = pd.DataFrame(
    [
        r.retrieval_metrics.to_dict()
        for r in results
    ]
)

print(metrics.mean())
metrics.mean().to_csv("retrieval_metrics.csv")

rows = []

for result in results:

    rows.append({

        "question": result.sample.question,

        "expected": result.sample.source_document,

        "retrieved_top1":
            result.result.retrieved_documents[0].document_id,

        **result.retrieval_metrics.to_dict(),
    })

pd.DataFrame(rows).to_csv(
    "retrieval_results.csv",
    index=False,
)

save_error_analysis(results)