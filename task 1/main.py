import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


load_dotenv()


def main():
    with open("./hypothesis_expansion.txt", "r") as f:
        prompt_template = f.read()

    prompt = PromptTemplate(
        input_variables=["max_query_length", "num_queries", "hypothesis"],
        template=prompt_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")

    chain = prompt | llm
    response = chain.invoke(
        input=dict(
            max_query_length=5,
            num_queries=10,
            hypothesis="Employees discussed adjusting, restricting, or reallocating energy schedules, bids, or load volumes in ways that could affect supply availability or market prices"
        )
    )
    print(response.content)


if __name__ == '__main__':
    main()