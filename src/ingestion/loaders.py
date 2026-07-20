from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Load a PDF into a list of LangChain Documents.

    Each page is returned as one Document.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} does not exist.")

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    return documents