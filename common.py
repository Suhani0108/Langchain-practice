"""
Shared helpers used by every example.

- get_model(): loads the API key from .env and creates the chat model
- run_safely(): runs a function and turns common errors into friendly messages
"""

import os
import sys

import groq
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Read variables from the .env file into the environment (os.environ).
load_dotenv()

# Groq model to use. It must support tool calling for the agent example.
# You can override it in .env with GROQ_MODEL=... (see README).
DEFAULT_MODEL = "openai/gpt-oss-120b"


class MissingAPIKeyError(Exception):
    """Raised when GROQ_API_KEY is not set."""


def get_model(temperature: float = 0):
    """Create a Groq chat model using LangChain's init_chat_model."""
    if not os.getenv("GROQ_API_KEY"):
        raise MissingAPIKeyError(
            "GROQ_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Paste your free key from https://console.groq.com/keys"
        )

    model_name = os.getenv("GROQ_MODEL", DEFAULT_MODEL)

    # "groq:<model>" tells init_chat_model which provider package to use
    # (langchain-groq). It reads GROQ_API_KEY from the environment itself.
    # max_retries=2 lets it retry briefly on a temporary error.
    return init_chat_model(
        f"groq:{model_name}", temperature=temperature, max_retries=2
    )


def friendly_error(error: Exception) -> str:
    """Turn an exception into a short, readable message."""
    if isinstance(error, MissingAPIKeyError):
        return str(error)
    if isinstance(error, groq.AuthenticationError):
        return "Your GROQ_API_KEY was rejected. Check the key in your .env file."
    if isinstance(error, groq.RateLimitError):
        return (
            "Groq rate limit reached (free tier). "
            "Wait a minute and try again."
        )
    if isinstance(error, groq.NotFoundError):
        return (
            "Groq could not find that model. Check GROQ_MODEL in .env against "
            "https://console.groq.com/docs/models"
        )
    if isinstance(error, groq.APIConnectionError):
        return "Could not reach Groq. Check your internet connection."
    if isinstance(error, groq.APIError):
        return f"Groq API error: {error}"
    return f"Unexpected error: {type(error).__name__}: {error}"


def run_safely(main) -> None:
    """Run main(), printing a friendly message instead of a traceback."""
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
    except Exception as error:
        print(f"\nError: {friendly_error(error)}")
        sys.exit(1)
