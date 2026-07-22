from pathlib import Path

from src.chains.llm import get_llm
from src.chains.prompts import get_rag_prompt
from src.chains.rag_chain import create_rag_chain

from src.retrieval.retrievers import create_retriever

from src.indexing.faiss_store import FAISSStore
from src.indexing.embeddings import get_embedding_model
from src.config import VECTORSTORE_DIR
from src.chains.rag_chain import format_docs
from langchain_core.output_parsers import StrOutputParser

import numpy as np


def main():

    print("=" * 60)
    print("Loading vector store...")
    print("=" * 60)

    embedding_model = get_embedding_model()
    print("✓ Embedding model loaded")

    store = FAISSStore()
    store.load(Path(VECTORSTORE_DIR), embedding_model=embedding_model)

    print(f"Loaded {store.size} vectors.")

    print("Creating retriever...")
    retriever = create_retriever(store.store)
    print("✓ Retriever created")

    prompt = get_rag_prompt()
    print("✓ Prompt created")

    
    llm = get_llm()
    print("✓ LLM created")


    print("Creating RAG chain...")
    pipeline = LangChainRAGPipeline(
    retriever=retriever,
    prompt=prompt,
    llm=llm,
)

    print("=" * 60)
    print("FinanceBench RAG")
    print("=" * 60)

    while True:

        question = input("\nQuestion (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        

        print("\nThinking...\n")

        result = pipeline.invoke(question)
        answer = result.prediction

        docs = retriever.invoke(question)
        
        print(f"Retrieved {len(docs)} documents\n")

        for i, doc in enumerate(docs):
            print("=" * 80)
            print(f"Document {i+1}")
            print(doc.metadata)
            print()
            print(doc.page_content[:1000])

        print(answer)
        print()


if __name__ == "__main__":
    main()