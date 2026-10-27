"""
02 - Prompt templates

Instead of building prompt strings by hand, a ChatPromptTemplate holds
placeholders like {topic} that get filled in at run time.

Run:  python 02_prompt_templates.py
"""

from langchain_core.prompts import ChatPromptTemplate

from common import get_model, run_safely


def main():
    model = get_model()

    # 1. A template with two variables: {level} and {topic}.
    #    Each tuple is (role, text). "system" = instructions, "human" = user.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a teacher. Explain things for a {level} audience."),
            ("human", "Explain {topic} in 2 sentences."),
        ]
    )

    # 2. Fill in the variables. The result is a list of ready-to-send messages.
    filled = prompt.invoke({"level": "10-year-old", "topic": "the internet"})
    print("Filled-in messages:")
    for message in filled.to_messages():
        print(f"  [{message.type}] {message.content}")

    # 3. Send them to the model.
    response = model.invoke(filled)
    print("\nAnswer:", response.content)

    # 4. Reuse the same template with different values.
    filled = prompt.invoke({"level": "university", "topic": "the internet"})
    print("\nUniversity-level answer:", model.invoke(filled).content)


if __name__ == "__main__":
    run_safely(main)
