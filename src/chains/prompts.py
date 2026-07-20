from langchain_core.prompts import ChatPromptTemplate


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template for the RAG pipeline.
    """

    template = """
You are a financial analyst.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, reply:
"I don't know."

Context:
{context}

Question:
{question}
"""

    return ChatPromptTemplate.from_template(template)