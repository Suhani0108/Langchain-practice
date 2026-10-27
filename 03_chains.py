"""
03 - Chains with the | (pipe) operator

LangChain Expression Language (LCEL) lets you connect steps with |.
The output of each step becomes the input of the next:

    prompt  ->  model  ->  output parser
    (dict)      (messages)  (AIMessage)   -> plain string

Run:  python 03_chains.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import get_model, run_safely


def main():
    model = get_model()

    # Step 1: a prompt template with one variable.
    prompt = ChatPromptTemplate.from_template(
        "Give me 3 fun facts about {animal}. Keep each fact to one line."
    )

    # Step 2: StrOutputParser pulls the text out of the AIMessage,
    # so the chain returns a plain string instead of a message object.
    parser = StrOutputParser()

    # Step 3: connect them into a chain.
    chain = prompt | model | parser

    # .invoke() runs the whole chain once.
    print("Octopus facts:\n", chain.invoke({"animal": "octopus"}))

    # .batch() runs it on several inputs (in parallel).
    results = chain.batch([{"animal": "penguin"}, {"animal": "owl"}])
    for animal, facts in zip(["penguin", "owl"], results):
        print(f"\n{animal.title()} facts:\n", facts)

    # .stream() prints the answer piece by piece as it is generated.
    print("\nStreaming a haiku:")
    haiku_chain = (
        ChatPromptTemplate.from_template("Write a haiku about {animal}.")
        | model
        | parser
    )
    for chunk in haiku_chain.stream({"animal": "cats"}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    run_safely(main)
