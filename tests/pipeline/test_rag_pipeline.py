from src.generation.prompt_builder import PromptBuilder
from src.retrieval.retriever import Retriever
from src.retrieval.models import RetrievalResult
from src.retrieval.reranker import Reranker
from src.generation.generator import Generator


class FakeRetriever(Retriever):
    
    def __init__(self):
        self.received_query = None

    def retrieve(self, query: str)-> list[RetrievalResult]:
        self.received_query = query
        return [
            RetrievalResult(
                document_id="doc_1",
                content="Apple reported record quarterly revenue.",
                score=0.92,
                metadata={"source": "10-K"},
            ),
            RetrievalResult(
                document_id="doc_2",
                content="Revenue increased by 8% year-over-year.",
                score=0.87,
                metadata={"source": "10-Q"},
            ),
        ]

class FakeReranker(Reranker):
    
    def __init__(self):
        self.received_query = None
        self.received_chunks = None
    

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        self.received_query = query
        self.received_chunks = chunks
        return chunks
    

class FakePromptBuilder(PromptBuilder):

    def __init__(self):
        self.received_question = None
        self.received_context = None

    def build(self, question, context):
        self.received_question = question
        self.received_context = context
        return "PROMPT"


class FakeGenerator(Generator):

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return "Generated answer"
    