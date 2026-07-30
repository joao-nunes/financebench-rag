from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from src.evaluation.metrics import AggregatedGenerationMetrics
from src.evaluation.models import EvaluationResult, EvaluationSample


@dataclass
class GenerationMetrics:
    context_recall: float
    faithfulness: float
    answer_correctness: float

    def to_dict(self):
        return {
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_correctness": self.answer_correctness,
        }


class GenerationEvaluator:

    def __init__(
        self,
        llm: BaseChatModel,
    ):
        self.judge = AggregatedGenerationMetrics(llm)

    def evaluate(
        self,
        sample: EvaluationSample,
        result: EvaluationResult,
    ) -> GenerationMetrics:

        scores = self.judge.compute(
            sample,
            result,
        )

        return GenerationMetrics(
            context_recall=scores.context_recall,
            faithfulness=scores.faithfulness,
            answer_correctness=scores.answer_correctness,
        )
