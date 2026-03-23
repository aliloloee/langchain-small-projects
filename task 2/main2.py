import os
from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_MODIFICATIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")

# react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_output_formatting = PromptTemplate(
    input_variables=["input", "tools", "agent_scratchpad"],
    template=REACT_PROMPT_WITH_MODIFICATIONS
).partial(format_instructions=output_parser.get_format_instructions())



agent = create_react_agent(
    llm=llm,
    tools=tools,           # Tools that was introduced to the agent (and internally added to the prompt)
    # prompt=react_prompt
    prompt=react_prompt_with_output_formatting
)

executer = AgentExecutor(
    agent=agent,
    tools=tools,           # Tools that executer have access to
    verbose=True
)

extract_output = RunnableLambda(lambda x: x['output'])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = executer | extract_output | parse_output


def main():
    result = chain.invoke(
        input = {
            "input": "Search for 3 papers written after 2021, about using retrieval augemented generation in forensics"
        }
    )
    print(result)


if __name__ == '__main__':
    main()