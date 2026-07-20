from src.ingestion.loaders import load_pdf
from src.ingestion.splitter import split_documents
from src.indexing.vectorstore import *
from src.indexing.embeddings import get_embedding_model
from src.retrieval.retrievers import create_retriever, retrieve


documents = load_pdf("data/financebench/pdfs/3M_2015_10K.pdf")

chunks = split_documents(documents)

embedding_model = get_embedding_model()

vectorstore = create_vectorstore(chunks, embedding_model)

save_vectorstore(vectorstore, "data/vectorstore")

vectorstore = load_vectorstore(
    "data/vectorstore",
    embedding_model,
)

retriever = create_retriever(vectorstore)

docs = retrieve(
    retriever,
    "What was Apple's revenue in 2022?"
)

print(len(docs))

print(docs[0].metadata)

print(docs[0].page_content[:500])