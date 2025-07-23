import csv

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool

neighborhood_data = []
with open('data_neighborhood.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        neighborhood_data.append(row)

header_translation = {
    "name": "Name",
    "apartment_number": "Apartment number",
    "phone": "Phone",
    "email": "Email",
    "owner": "Owner",
    "last_payment": "Last payment",
    "hobbies": "Hobbies"
}

docs = [
    Document(
        "\n".join(f"{header_translation[key]}: {value}" for key, value in neighboor.items()),
        metadata={"name": neighboor["name"]}
    )
    for neighboor in neighborhood_data
]

retriever = BM25Retriever.from_documents(docs)

def extract_info(query: str) -> str:
    """Extract information from the neighborhood data based on the query.
    
    Args:
        query: The query to search for in the neighborhood data.

    Returns:
        str: The neighborhood data that matches the query.
    """
    retrieved_docs = retriever.invoke(query)

    if retrieved_docs:
        return "\n\n".join([doc.page_content for doc in retrieved_docs])
    else:
        return "No results found for the query."

neighborhood_info_tool = Tool(
    name="neighborhood_info_tool",
    description="Use this tool if you need information about the neighbors and the neighborhood",
    func=extract_info
)
