"""A SCRIPT FOR PREPARING SQUAD DATASET FOR QUESTION ANSWER GENERATION TASK, ANSWER AWARE."""
import datasets

TOKEN_QUESTION = "<question>"
TOKEN_END_QUESTION = "</question>"
TOKEN_CONTEXT = "<context>"
TOKEN_END_CONTEXT = "</context>"
TOKEN_ANSWER = "<answer>"
TOKEN_END_ANSWER = "</answer>"
SPLIT_SEED = 42
NPROC = 32


def preprocess_squad_dataset(dataset_name="squad", split="train"):
    """Preprocesses the dataset for the answer aware question generation task.
    Returns a dataset with the following columns:

        formatted_text: The text with the question, answer and context tokens added.
        id: The id of the example.
        title: The title of the example.
        validation: The validation split of the dataset.
        train: The train split of the dataset.

    Args:
        dataset_name (str, optional): The name of the dataset to load. Defaults to 'squad'.
        split (str, optional): The split of the dataset to load. Defaults to 'train'.

    Returns:
        datasets.DatasetDict: The dataset with the formatted_text column added.
    """
    dataset = datasets.load_dataset(dataset_name, split=split)
    # Add question, answer and context tokens to dataset in a new column named text
    dataset = dataset.map(
        lambda e: {
            "formatted_text": f'{TOKEN_QUESTION} {e["question"]} {TOKEN_END_QUESTION} {TOKEN_CONTEXT} {e["context"]} {TOKEN_END_CONTEXT} {TOKEN_ANSWER} {e["answers"]["text"][0]} {TOKEN_END_ANSWER}'
        },
        num_proc=NPROC,
    )

    # Remove unnecessary columns, leaving only the formatted_text column
    dataset = dataset.remove_columns(["answers", "context", "question"])

    if split == "train":
        dataset = dataset.train_test_split(test_size=0.1, seed=SPLIT_SEED)
        dataset["validation"] = dataset.pop("test")

    return dataset
