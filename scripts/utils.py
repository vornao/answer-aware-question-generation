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

