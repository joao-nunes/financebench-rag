class RAGService:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def answer(self, question: str):
        return self.pipeline.answer(question)