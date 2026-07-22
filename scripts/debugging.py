from src.ingestion.loaders import load_pdf
from src.ingestion.splitter import split_documents

pdf_path = "./data/financebench/pdfs/APPLE_2022_10K.pdf"

documents = load_pdf(pdf_path)
chunks = split_documents(documents)

print(chunks[0].metadata)