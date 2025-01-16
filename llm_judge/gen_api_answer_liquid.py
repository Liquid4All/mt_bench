"""Generate answers with GPT-4

Usage:
python3 gen_api_answer.py --model lfm-3b-jp --openai-api-key YOUR_API_KEY --openai-api-base https://inference-1.liquid.ai/v1
"""
import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime

import openai
import shortuuid
import tqdm

from llm_judge.common import (
    load_questions,
    chat_completion_openai,
)
from llm_judge.gen_model_answer import reorg_answer_file
from model.model_adapter import get_conversation_template


def get_answer(
    question: dict, model: str, num_choices: int, max_tokens: int, answer_file: str
):
    choices = []
    for i in range(num_choices):
        conv = get_conversation_template(model)

        turns = []
        for j in range(len(question["turns"])):
            conv.append_message(conv.roles[0], question["turns"][j])
            conv.append_message(conv.roles[1], None)

            output = chat_completion_openai(model, conv, temperature=None, max_tokens=max_tokens)

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding='utf-8') as fout:
        fout.write(json.dumps(ans, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument("--openai-api-key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument(
        "--openai-api-base",
        type=str,
        required=True,
        help="Custom OpenAI API endpoint base URL.",
    )
    args = parser.parse_args()

    # Print information about the configuration
    print(f"Using model: {args.model}")
    if "liquid.ai" in args.openai_api_base:
        print("Using LiquidAI API.")
    elif "localhost" in args.openai_api_base:
        print("Using on-prem API.")
    else:
        print("Using external API.")

    # Configure OpenAI API
    openai.api_key = args.openai_api_key
    openai.api_base = args.openai_api_base

    current_dir = os.path.dirname(os.path.abspath(__file__))
    question_file = os.path.join(current_dir, "data", args.bench_name, "question.jsonl")
    questions = load_questions(question_file, args.question_begin, args.question_end)

    if args.answer_file:
        answer_file = args.answer_file
    else:
        now = datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        answer_file = os.path.join(current_dir, "data", args.bench_name, "model_answer", f"{args.model}-{timestamp}.jsonl")
    print(f"Output to {answer_file}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            future = executor.submit(
                get_answer,
                question,
                args.model,
                args.num_choices,
                args.max_tokens,
                answer_file,
            )
            futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()

    reorg_answer_file(answer_file)
