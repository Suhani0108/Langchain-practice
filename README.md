# langchain-practice

A hands-on walkthrough of **LangChain 1.x**, built while learning it. Each numbered file adds one new idea. It goes from a single model call to a tool-using research agent with a chat UI.

Everything here is **free**:

| Piece | What it uses | Cost |
|---|---|---|
| LLM | [Groq](https://groq.com) free tier, `openai/gpt-oss-120b` | Free (rate-limited) |
| Web search | DuckDuckGo via the `ddgs` package | Free, no key |
| Encyclopedia | Wikipedia API | Free, no key |
| UI | Streamlit | Free |

## What each file teaches

| File | Concept |
|---|---|
| [`01_first_llm_call.py`](01_first_llm_call.py) | `init_chat_model`, `.invoke()`, `SystemMessage` / `HumanMessage`, reading `.content` and token usage |
| [`02_prompt_templates.py`](02_prompt_templates.py) | `ChatPromptTemplate` with `{variables}`, reusing one template with different inputs |
| [`03_chains.py`](03_chains.py) | LCEL chains: `prompt \| model \| StrOutputParser()`, plus `.invoke()`, `.batch()` and `.stream()` |
| [`04_output_parsers.py`](04_output_parsers.py) | Structured output as a Pydantic object, using both `with_structured_output()` and `PydanticOutputParser` |
| [`05_search_agent.py`](05_search_agent.py) | An agent built with `create_agent`: tools, a ReAct loop, conversation memory, and an interactive terminal chat that shows which tools were used |
| [`agent_setup.py`](agent_setup.py) | The agent's 4 tools (made with `@tool`), its system prompt, and its memory (`InMemorySaver`). Shared by `05` and `app.py` |
| [`app.py`](app.py) | A Streamlit chat UI for the same agent |
| [`common.py`](common.py) | Loads `.env`, creates the model, and turns errors into friendly messages |

### The agent's tools

- **`web_search`**: DuckDuckGo results for current or recent information
- **`wikipedia_lookup`**: a short Wikipedia summary for established facts
- **`calculator`**: safe math evaluation that does not use `eval()`. Supports `+ - * / ** %`, `sqrt`, `log`, `sin`, `pi`, and more
- **`get_today_date`**: today's date, so the model doesn't guess it

## Get a free Groq API key

1. Go to <https://console.groq.com> and sign up. No credit card is needed.
2. Open **API Keys** (<https://console.groq.com/keys>) and click **Create API Key**.
3. Copy the key, which starts with `gsk_`. You won't be able to see it again.

## Setup

Requires **Python 3.10+**. LangChain 1.x does not support 3.9.

```bash
git clone <your-repo-url> langchain-practice
cd langchain-practice

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your key
cp .env.example .env              # then open .env and paste your key after GROQ_API_KEY=
```

`.env` is in `.gitignore`, so your key is never committed.

## Run

```bash
python 01_first_llm_call.py
python 02_prompt_templates.py
python 03_chains.py
python 04_output_parsers.py
python 05_search_agent.py         # interactive chat: type 'new' to reset, 'quit' to exit

streamlit run app.py              # opens the chat UI in your browser
```

### Example terminal session

```
You: What is 17% of 2,340?

Agent: 17% of 2,340 is 397.8.
[Tools used: calculator]
```

## Example questions to try

Each question tends to trigger a different tool:

- *"What's the latest news about NASA?"* → `web_search`
- *"Who was Ada Lovelace?"* → `wikipedia_lookup`
- *"What is sqrt(2025) + 15% of 480?"* → `calculator`
- *"What day of the week is it today?"* → `get_today_date`
- *"How many days are left until Christmas?"* → `get_today_date` + `calculator`
- *"When was the Eiffel Tower built, and how many years ago was that?"* → `wikipedia_lookup` + `get_today_date` + `calculator`

To test memory, ask a follow-up question:

- *"Who founded Microsoft?"* then *"How old is he now?"*

## How the agent picks a tool (the ReAct loop)

`create_agent` runs a **Reason + Act** loop:

```
          ┌────────────────────────────────────────────┐
question ─▶  MODEL: read the conversation + tool list   │
          │  Need a tool? ── no ──▶ final answer ───────┼─▶ done
          │      │ yes                                  │
          │      ▼                                      │
          │  emit a tool call, e.g.                     │
          │  calculator(expression="17*23")             │
          │      │                                      │
          │      ▼                                      │
          │  TOOL runs → result added to conversation ──┘ (loop)
          └────────────────────────────────────────────┘
```

1. **The model sees each tool's description.** `@tool` turns the function name, its type hints, and its docstring into a schema. That schema is sent to the model with every request, and the docstring is the main thing the model uses to choose a tool.
2. **The model decides.** Instead of text, it can reply with a *tool call*: the tool's name plus JSON arguments. It may request several tools at once.
3. **LangChain runs the tool** and adds the result to the conversation as a `ToolMessage`.
4. **The model looks again**, now with the tool result in view. It can call more tools, or it can write the final answer. The loop ends when the model replies without any tool calls.

The system prompt in [`agent_setup.py`](agent_setup.py) adds guidance, such as "always use calculator for arithmetic". If a tool fails, it returns an error message instead of crashing, so the model can try a different tool.

**Memory:** `InMemorySaver` stores each conversation under a `thread_id`. When you send a new message with the same `thread_id`, the agent loads the earlier messages first. That is how follow-up questions work. The memory is cleared when the program exits.

## Error handling

| Problem | What you'll see |
|---|---|
| `GROQ_API_KEY` missing | Steps for creating `.env` |
| Invalid key | "Your GROQ_API_KEY was rejected…" |
| Groq rate limit (free tier) | "Groq rate limit reached… wait a minute". The chat keeps running |
| DuckDuckGo / Wikipedia failure | The tool returns an error message to the agent, and the agent tries another tool |
| Retired model | "Groq could not find that model". Set `GROQ_MODEL` in `.env` |

## Changing the model

Groq retires models from time to time. For example, `llama-3.3-70b-versatile` was retired from the free tier in August 2026. To use a different tool-calling model, add it to `.env`:

```
GROQ_MODEL=openai/gpt-oss-20b
```

Current models are listed at <https://console.groq.com/docs/models>. Tool support is listed at <https://console.groq.com/docs/tool-use>.
