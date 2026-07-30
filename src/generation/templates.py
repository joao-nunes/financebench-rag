RAG_PROMPT = """
Answer the user's question using only the provided context.

If the answer cannot be found in the context, reply:
"I don't know based on the provided context."

Context
-------
{context}

Question
--------
{question}

Answer
------
""".strip()
