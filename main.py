from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent reponse with answer and sources"""

    answer: str = Field(description="The Agent answer to the query")
    sources: list[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


llm = ChatOpenAI(model="gpt-5")

search_tool = TavilySearch()
tools = [search_tool]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


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
