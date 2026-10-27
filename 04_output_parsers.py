"""
04 - Structured output with Pydantic

Free text is hard for code to use. Here we ask the model to fill in a
Pydantic model so we get a real Python object with typed fields.

Two ways are shown:
  A. model.with_structured_output(...)  - the modern, recommended way
  B. PydanticOutputParser in a chain    - the classic "output parser" way

Run:  python 04_output_parsers.py
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from common import get_model, run_safely


# The shape we want back. Field descriptions are sent to the model,
# so they act as instructions for each field.
class Movie(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Year the movie was released")
    genres: list[str] = Field(description="1 to 3 genres")
    rating: float = Field(description="Your rating out of 10")
    summary: str = Field(description="One-sentence plot summary")


def main():
    model = get_model()

    # --- A. with_structured_output -------------------------------------
    # The model is told to answer using the Movie schema, and LangChain
    # converts the reply into a Movie object for us.
    structured_model = model.with_structured_output(Movie)
    movie = structured_model.invoke("Tell me about the movie Inception.")

    print("A. with_structured_output")
    print("   type:", type(movie).__name__)
    print("   title:", movie.title)
    print("   year:", movie.year, "| genres:", ", ".join(movie.genres))
    print("   rating:", movie.rating)
    print("   summary:", movie.summary)

    # --- B. PydanticOutputParser ---------------------------------------
    # The parser writes format instructions (a JSON schema) that we put
    # into the prompt, then parses the model's JSON reply into a Movie.
    parser = PydanticOutputParser(pydantic_object=Movie)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer only with JSON.\n{format_instructions}"),
            ("human", "Tell me about the movie {movie_name}."),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    # .partial() fills one variable now, so we only pass movie_name later.

    chain = prompt | model | parser
    movie = chain.invoke({"movie_name": "Spirited Away"})

    print("\nB. PydanticOutputParser")
    # model_dump() turns the Pydantic object into a normal dict.
    print("  ", movie.model_dump())


if __name__ == "__main__":
    run_safely(main)
