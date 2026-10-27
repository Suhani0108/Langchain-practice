"""
The search agent's tools and setup.

Shared by 05_search_agent.py (terminal chat) and app.py (Streamlit UI).
Read this file together with 05_search_agent.py.
"""

import ast
import math
import operator
from datetime import datetime

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from common import get_model

# ---------------------------------------------------------------------------
# Tools
#
# @tool turns a normal Python function into a LangChain tool.
# The function NAME, the TYPE HINTS and the DOCSTRING are sent to the model.
# The model reads the docstring to decide when to use the tool, so make it
# clear and specific.
#
# If a tool fails, it returns an error message instead of crashing. The
# model sees that message and can retry or try another tool.
# ---------------------------------------------------------------------------


@tool
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Use for current events, news,
    recent facts, or anything that may have changed recently."""
    try:
        results = DDGS().text(query, max_results=5)
    except DDGSException as error:  # includes rate limits and timeouts
        return f"Web search failed ({type(error).__name__}). Try again or use another tool."

    if not results:
        return "No web results found."

    # Format each result as: title, link, snippet.
    return "\n\n".join(
        f"{r['title']}\n{r['href']}\n{r['body']}" for r in results
    )


# Wikipedia blocks requests that don't send a User-Agent header.
WIKI_HEADERS = {"User-Agent": "langchain-practice/1.0 (learning project)"}


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia and return a short summary. Use for
    well-established facts about people, places, history, and science."""
    try:
        # Step 1: search for the best-matching article title.
        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "srlimit": 1,
                "format": "json",
            },
            headers=WIKI_HEADERS,
            timeout=10,
        )
        search.raise_for_status()
        hits = search.json()["query"]["search"]
        if not hits:
            return f"No Wikipedia article found for '{topic}'."
        title = hits[0]["title"]

        # Step 2: get that article's summary.
        summary = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + title.replace(" ", "_"),
            headers=WIKI_HEADERS,
            timeout=10,
        )
        summary.raise_for_status()
        return f"{title}: {summary.json().get('extract', 'No summary available.')}"
    except requests.RequestException as error:
        return f"Wikipedia lookup failed ({type(error).__name__}). Try web_search instead."


# The operators and functions the calculator allows. Anything else is rejected,
# which makes this much safer than Python's eval().
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
}
CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node):
    """Walk the parsed expression tree and compute the result."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in CONSTANTS:
        return CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        # Huge powers like 9**9**9 would freeze the program, so limit them.
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_evaluate(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FUNCTIONS
    ):
        return FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '(15 * 4) / 3' or 'sqrt(144) + 2**10'.
    Supports + - * / // % **, parentheses, sqrt, log, sin, cos, tan, abs,
    round, pi and e. Use this for any arithmetic instead of guessing."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError, TypeError, OverflowError) as error:
        return f"Error: could not calculate '{expression}' ({error})."


@tool
def get_today_date() -> str:
    """Return today's date and the current local time. Use when the question
    depends on today's date, e.g. 'how many days until...' or 'how old is...'."""
    return datetime.now().strftime("Today is %A, %B %d, %Y. Local time is %H:%M.")


TOOLS = [web_search, wikipedia_lookup, calculator, get_today_date]

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful research assistant.
- Use web_search for recent or changing information.
- Use wikipedia_lookup for established background facts.
- Always use calculator for arithmetic. Never do math in your head.
- Use get_today_date whenever the answer depends on today's date.
- If a tool fails, try another tool or say what went wrong.
Keep answers short and mention your sources when you used search tools."""


def build_agent():
    """Create the agent. create_agent builds the ReAct loop for us:
    model -> (tool calls -> tools -> model)* -> final answer."""
    return create_agent(
        model=get_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        # The checkpointer saves the conversation after each turn, so the
        # agent remembers earlier messages that used the same thread_id.
        checkpointer=InMemorySaver(),
    )


def ask(agent, question: str, thread_id: str):
    """Send one question to the agent.

    Returns (answer_text, list_of_tools_used).
    """
    # thread_id picks which saved conversation to continue.
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config,
    )
    messages = result["messages"]  # the WHOLE conversation so far

    # Only look at messages from this turn (after the last HumanMessage).
    last_human = max(
        i for i, m in enumerate(messages) if isinstance(m, HumanMessage)
    )
    this_turn = messages[last_human + 1 :]

    # Each AIMessage may contain tool_calls: the tools the model asked to run.
    tools_used = [
        call["name"]
        for message in this_turn
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]

    answer = messages[-1].text  # the final AIMessage is the answer
    return answer, tools_used
