import re

def enrich_chunk(document):
    """
    Prepend document metadata to the chunk text before embedding.
    """

    document_id = document.metadata["document_id"]

    # Example: APPLE_2022_10K
    match = re.match(r"(.+?)_(\d{4}(?:Q\d)?)_(10-K|10Q|10-Q|8-K|10K)", document_id)

    if match:
        company = match.group(1).replace("_", " ")
        period = match.group(2)
        filing = match.group(3).replace("10K", "10-K").replace("10Q", "10-Q")
    else:
        company = "Unknown"
        period = "Unknown"
        filing = "Unknown"

    header = (
        f"Company: {company}\n"
        f"Period: {period}\n"
        f"Filing: {filing}\n"
        f"Document ID: {document_id}\n\n"
    )

    document.page_content = header + document.page_content

    return document