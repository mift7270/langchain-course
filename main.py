from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Args:
        query:  Tool that searches over internet

    Returns:
        The Search result
    """
    print(f"Searching the internet for: {query}")

    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain course")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search for 3 job postings for Remote Data Analytics Managers roles in linkedin and list their details"
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
