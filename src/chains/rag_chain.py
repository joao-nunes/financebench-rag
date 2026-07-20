from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel


def format_docs(docs: list[Document]) -> str:
    """
    Convert a list of retrieved Documents into a single context string.

    Parameters
    ----------
    docs : list[Document]
        Documents returned by the retriever.

    Returns
    -------
    str
        Concatenated document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(
    retriever,
    prompt: ChatPromptTemplate,
    llm: BaseChatModel,
):
    """
    Create the baseline LCEL Retrieval-Augmented Generation (RAG) pipeline.

    Pipeline:
        Question
            ↓
        Retriever
            ↓
        Format retrieved documents
            ↓
        Prompt template
            ↓
        Language model
            ↓
        String output parser

    Parameters
    ----------
    retriever
        LangChain retriever.

    prompt : ChatPromptTemplate
        Prompt template used to instruct the LLM.

    llm : BaseChatModel
        Chat language model.

    Returns
    -------
    Runnable
        Executable LCEL RAG chain.
    """

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
