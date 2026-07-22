from src.evaluation.dataset import FinanceBenchDataset

dataset = FinanceBenchDataset(
    "data/financebench/data/financebench_open_source.jsonl"
)

samples = dataset.load()

print(f"Loaded {len(samples)} questions")

sample = samples[0]

print(sample.question)
print(sample.reference_answer)
print(sample.reference_documents)
print(sample.metadata)