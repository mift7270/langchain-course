import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Initialize the embeddings model (same as ingestion.py)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore(
    index_name="langchain-docs-index", embedding=embeddings
)

# initialize chat model

model = init_chat_model("gpt-5.2", model_provider="openai")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain"""

    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to answer a query using a retrieved documentation

    Args:
      query: The user's question

    Returns:
      Dictinoary containint:
        -answer: The generted answer
        -contect: List of retrieved documents
    """

    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentaion. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sourcess you use in your answers. "
        "if you cannot find the naswer in the retrived documantation, say so. "
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list

    messages = [{"role": "user", "content": query}]

    response = agent.invoke({"messages": messages})

    # extract the answer from the last AI message

    answer = response["messages"][-1].content

    # extract context document from ToolMessage artifacts

    context_docs = []

    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    result = run_llm(query="What are deep agents")
    print(result)
