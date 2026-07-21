from pathlib import Path

from src.chains.llm import get_llm
from src.chains.prompts import get_rag_prompt
from src.chains.rag_chain import create_rag_chain

from src.retrieval.retrievers import create_retriever

from src.indexing.vectorstore import load_vectorstore
from src.indexing.embeddings import get_embedding_model
from src.config import VECTORSTORE_DIR


def main():

    print("=" * 60)
    print("Loading vector store...")
    print("=" * 60)


    embedding_model = get_embedding_model()
    print("✓ Embedding model loaded")

    store = load_vectorstore(
        Path(VECTORSTORE_DIR),
        embedding_model,
    )

    print(f"Loaded {store.size} vectors.")

    retriever = create_retriever(store.store)
    print(f"✓ Loaded {store.size} vectors")

    print(f"Embedding dimension: {len(embedding_model.embed_query('hello'))}")

    print("Creating retriever...")
    retriever = create_retriever(store.store)
    print("✓ Retriever created")

    prompt = get_rag_prompt()

    llm = get_llm()

    rag_chain = create_rag_chain(
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

        answer = rag_chain.invoke(question)

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