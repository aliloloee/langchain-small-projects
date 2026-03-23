import os
from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
llm = ChatOpenAI(model="gpt-4o")

agent = create_react_agent(
    llm=llm,
    tools=tools,           # Tools that was introduced to the agent (and internally added to the prompt)
    prompt=react_prompt
)

executer = AgentExecutor(
    agent=agent,
    tools=tools,           # Tools that executer have access to
    verbose=True
)
chain = executer


def main():
    result = chain.invoke(
        input = {
            "input": "Search for 3 papers written after 2021, about using retrieval augemented generation in forensics"
        }
    )
    print(result)


if __name__ == '__main__':
    main()