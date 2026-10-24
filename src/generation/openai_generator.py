import logging
import os

from openai import OpenAI

from src.exceptions import GenerationError
from src.generation.generator import Generator
from src.observability import observation

logger = logging.getLogger(__name__)


class OpenAIGenerator(Generator):

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
    ):
        api_key = api_key or os.environ["OPENAI_API_KEY"]

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:

        with observation(
            "generation", as_type="generation", model=self.model, input=prompt
        ) as generation:
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=prompt,
                )
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                raise GenerationError("Failed to generate response.") from e

            generation.update(output=response.output_text)
            if response.usage is not None:
                generation.update(
                    usage_details={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    }
                )
            return response.output_text
