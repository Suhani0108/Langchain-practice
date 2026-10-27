"""
01 - First LLM call

The simplest possible LangChain program: send a message, print the reply.

Run:  python 01_first_llm_call.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from common import get_model, run_safely


def main():
    # 1. Create the chat model (Groq, via init_chat_model in common.py).
    model = get_model()

    # 2. Easiest way: pass a plain string. LangChain wraps it in a HumanMessage.
    response = model.invoke("In one sentence, what is LangChain?")

    # The reply is an AIMessage. Its text lives in .content
    print("Answer:", response.content)

    # 3. Chat models really work with a list of messages.
    #    A SystemMessage sets the model's behaviour; a HumanMessage is the user.
    messages = [
        SystemMessage("You are a pirate. Answer in one short sentence."),
        HumanMessage("What is Python?"),
    ]
    response = model.invoke(messages)
    print("\nPirate answer:", response.content)

    # 4. The response also carries metadata, such as token usage.
    print("\nToken usage:", response.usage_metadata)


if __name__ == "__main__":
    run_safely(main)
