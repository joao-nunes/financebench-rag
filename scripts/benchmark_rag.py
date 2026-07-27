from pathlib import Path

import pandas as pd

from src.chains.llm import get_llm
from src.chains.prompts import get_rag_prompt
from src.chains.rag_chain import create_generation_chain
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.dataset import FinanceBenchDataset
from src.evaluation.generation import GenerationEvaluator
from src.evaluation.pipeline import FinanceBenchRAGPipeline
from src.evaluation.splits import Split
from src.evaluation.writer import ExperimentWriter
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.retrieval.reranking import CrossEncoderReranker
from src.retrieval.retrievers import create_retriever

EXPERIMENT_NAME = "baseline_rag"

VECTORSTORE_PATH = Path("./data/vectorstore")
DATASET_PATH = Path("./data/financebench/data/financebench_open_source.jsonl")
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_NAME

embedding_model = get_embedding_model()

vectorstore = FAISSStore()
vectorstore.load(
    VECTORSTORE_PATH,
    embedding_model,
)

retriever = create_retriever(
    vectorstore.store,
)

reranker = CrossEncoderReranker()

generator = create_generation_chain(
    prompt=get_rag_prompt(),
    llm=get_llm(),
)

pipeline = FinanceBenchRAGPipeline(
    retriever=retriever,
    reranker=reranker,
    generator=generator,
)

llm = get_llm()

generation_evaluator = GenerationEvaluator(
    llm=llm,
)

runner = BenchmarkRunner(
    rag_pipeline=pipeline,
    evaluator=generation_evaluator,
)

dataset = FinanceBenchDataset.from_jsonl(DATASET_PATH)
split = Split.load("data/financebench/splits")
dataset = dataset.subset(split.train_ids)

results = runner.evaluate(dataset)

writer = ExperimentWriter(EXPERIMENT_DIR)

writer.save_benchmark_results(results)

metrics = pd.DataFrame(
    [
        r.metrics.to_dict()
        for r in results
    ]
)

writer.save_metrics(
    metrics.mean().to_dict()
)

writer.save_environment()

print(metrics.mean())