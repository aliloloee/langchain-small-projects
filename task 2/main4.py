import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")

# Create agent with structured output using the new create_agent API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful AI assistant with access to search tools. Answer questions thoroughly and provide relevant sources.",
    response_format=ProviderStrategy(AgentResponse),
)


def main():
    """
    Migrated to use the new create_agent API with structured output.
    The agent now uses LangGraph internally for better performance and control.
    """
    result = agent.invoke(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?",
                    # "content": "search for 3 papers written after 2021, about using retrieval augemented generation in forensic analysis",
                }
            ]
        }
    )

    # The structured_response is automatically extracted when using response_format
    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
