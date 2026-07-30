from openai import OpenAI

from src.generation.generator import Generator
from src.exceptions import GenerationError

import os
from openai import OpenAI

import logging

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

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
            )
            return response.output_text
        except Exception as e:
            raise GenerationError("Failed to generate response.") from e
