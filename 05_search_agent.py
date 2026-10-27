"""
05 - Search agent (interactive terminal chat)

An agent is an LLM that can decide to call tools. This one has:
  - web_search        (DuckDuckGo, no API key)
  - wikipedia_lookup  (Wikipedia API)
  - calculator        (written with @tool)
  - get_today_date    (written with @tool)

The tools and agent are defined in agent_setup.py. This file runs the chat.
The agent remembers the conversation, so follow-up questions work.

Run:  python 05_search_agent.py
"""

import uuid

from agent_setup import TOOLS, ask, build_agent
from common import MissingAPIKeyError, friendly_error


def main():
    try:
        agent = build_agent()
    except MissingAPIKeyError as error:
        print(f"Error: {error}")
        return

    # One thread_id = one conversation. A new id would start fresh.
    thread_id = str(uuid.uuid4())

    print("Search agent ready! Tools:", ", ".join(t.name for t in TOOLS))
    print("Type a question, 'new' to start over, or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if question.lower() == "new":
            thread_id = str(uuid.uuid4())
            print("(Started a new conversation.)\n")
            continue

        try:
            answer, tools_used = ask(agent, question, thread_id)
        except KeyboardInterrupt:
            print("\n(Cancelled.)\n")
            continue
        except Exception as error:
            # Rate limits, network problems, etc.: report and keep chatting.
            print(f"Error: {friendly_error(error)}\n")
            continue

        print(f"\nAgent: {answer}")
        print(f"[Tools used: {', '.join(tools_used) if tools_used else 'none'}]\n")


if __name__ == "__main__":
    main()
