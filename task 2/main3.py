import os
from dotenv import load_dotenv

from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_MODIFICATIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
structure_results_llm = llm.with_structured_output(AgentResponse)


react_prompt_with_output_formatting = PromptTemplate(
    input_variables=["input", "tools", "agent_scratchpad"],
    template=REACT_PROMPT_WITH_MODIFICATIONS
).partial(format_instructions="")



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


# Less tokens passed to 1st LLM call, but there will 1 more LLM call in the end for structuring the outpu
chain = executer | extract_output | structure_results_llm


def main():
    """
    The previous method in "main2.py", worked much more consistent with input
    """
    result = chain.invoke(
        input = {
            # "input": "Search for 3 papers written after 2021, about using retrieval augemented generation in forensics"
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
        }
    )
    print(result)


if __name__ == '__main__':
    main()