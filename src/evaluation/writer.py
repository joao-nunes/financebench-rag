from pathlib import Path
import json
from dataclasses import asdict
from src.evaluation.models import BenchmarkResult
import csv
import platform
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


class ExperimentWriter:

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir

    
    def save_metrics(
        self,
        metrics: dict,
    ):
        output_path = self.experiment_dir / "metrics.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
    
    def save_benchmark_results(
        self,
        results: list[BenchmarkResult],
    ):
        output_path = self.experiment_dir / "benchmark_results.jsonl"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:

            for result in results:

                f.write(
                    json.dumps(
                        asdict(result)
                    )
                )

                f.write("\n")

    def save_retrieval_failures(
        self,
        benchmark_results: list[BenchmarkResult],
    ) -> None:
        """
        Save retrieval failures to a CSV file.

        A failure is defined as the expected document not being retrieved
        by the retriever (i.e. unbounded recall == 0).

        Columns
        -------
        question
        expected_document
        retrieved_documents
        """
        output_path = self.experiment_dir / "retrieval_failures.jsonl"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "question",
                    "expected_document",
                    "retrieved_documents",
                ]
            )

            n_failures = 0

            for result in benchmark_results:

                if result.retrieval_metrics.unbounded_recall == 1.0:
                    continue

                writer.writerow(
                    [
                        result.sample.question,
                        result.sample.source_document,
                        " | ".join(
                            doc.document_id
                            for doc in result.result.retrieved_documents
                        ),
                    ]
                )

                n_failures += 1

        print(f"Saved {n_failures} retrieval failures.")

    
    def save_environment(
        self,
    ) -> None:
        """
        Save the software environment used for the experiment.
        """

        output_path = self.experiment_dir / "environment.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        environment = {
            "timestamp": datetime.now().isoformat(),

            "python": {
                "version": sys.version,
                "implementation": platform.python_implementation(),
            },

            "system": {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },

            "git": {
                "commit": _git_commit(),
            },

            "packages": {
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "sentence-transformers": _package_version("sentence-transformers"),
                "langchain": _package_version("langchain"),
                "langchain-community": _package_version("langchain-community"),
                "faiss-cpu": _package_version("faiss-cpu"),
                "faiss-gpu": _package_version("faiss-gpu"),
                "openai": _package_version("openai"),
                "numpy": _package_version("numpy"),
                "pandas": _package_version("pandas"),
            },
            "command": " ".join(sys.argv)
        }

        with output_path.open("w") as f:
            json.dump(environment, f, indent=4)