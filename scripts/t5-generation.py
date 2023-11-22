"""Generate questions given a context using T5."""

import argparse
from transformers import T5ForConditionalGeneration, T5TokenizerFast

TOKEN_QUESTION = "<question>"
TOKEN_END_QUESTION = "</question>"
TOKEN_CONTEXT = "<context>"
TOKEN_END_CONTEXT = "</context>"
TOKEN_ANSWER = "<answer>"
TOKEN_END_ANSWER = "</answer>"


def generate_questions(
    context, model_name="t5-small", max_length=32, num_beams=4, num_return_sequences=4
):
    """Generate questions given a context using T5.

    Args:
        context (str): The context to generate questions for.
        model_name (str, optional): The name of the model to use. Defaults to 't5-small'.
        max_length (int, optional): The maximum length of the generated question. Defaults to 32.
        num_beams (int, optional): The number of beams to use for beam search. Defaults to 4.
        num_return_sequences (int, optional): The number of sequences to return. Defaults to 4.

    Returns:
        list: The list of generated questions.
    """
    # Load tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Encode context
    input_ids = tokenizer.encode(context, return_tensors="pt")

    # Generate questions
    questions = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
    )

    # Decode questions
    questions = tokenizer.batch_decode(questions, skip_special_tokens=True)

    return questions


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate questions given a context using T5."
    )
    parser.add_argument(
        "--context", type=str, help="The context to generate questions for."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="t5-small",
        help="The name of the model to use.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="The maximum length of the generated question.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="The number of beams to use for beam search.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=4,
        help="The number of sequences to return.",
    )
    args = parser.parse_args()

    # Generate questions
    questions = generate_questions(
        context="generate question: <answer> Denver Broncos <answer> <context> Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the golden anniversary with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as Super Bowl), so that the logo could prominently feature the Arabic numerals 50. <context>",
        model_name=args.model_name,
        max_length=32,
        num_return_sequences=1,
    )

    # Print questions
    for question in questions:
        print(question)
