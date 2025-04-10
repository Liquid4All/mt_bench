"""
Usage:
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --mode [single|pairwise-baseline|pairwise-all]
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)


def make_match(
    questions,
    models,
    model_answers,
    judge,
    baseline_model,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m_1 = models[i]
            m_2 = baseline_model
            if m_1 == m_2:
                continue
            a_1 = model_answers[m_1][q_id]
            a_2 = model_answers[baseline_model][q_id]
            if ref_answers is not None:
                ref = ref_answers['gpt-4'][q_id]
                match = MatchPair(
                    dict(q),
                    m_1,
                    m_2,
                    a_1,
                    a_2,
                    judge,
                    ref_answer=ref,
                    multi_turn=multi_turn,
                )
            else:
                match = MatchPair(
                    dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                )
            matches.append(match)
    return matches


def make_match_all_pairs(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                q_id = q["question_id"]
                m_1 = models[i]
                m_2 = models[j]
                a_1 = model_answers[m_1][q_id]
                a_2 = model_answers[m_2][q_id]
                if ref_answers is not None:
                    ref = ref_answers['gpt-4'][q_id]
                    match = MatchPair(
                        dict(q),
                        m_1,
                        m_2,
                        a_1,
                        a_2,
                        judge,
                        ref_answer=ref,
                        multi_turn=multi_turn,
                    )
                else:
                    match = MatchPair(
                        dict(q), m_1, m_2, a_1, a_2, judge, multi_turn=multi_turn
                    )
                matches.append(match)
    return matches


def make_match_single(
    questions,
    models,
    model_answers,
    judge,
    baseline_model=None,
    ref_answers=None,
    multi_turn=False,
):
    matches = []
    for q in questions:
        if multi_turn and len(q["turns"]) != 2:
            continue
        for i in range(len(models)):
            q_id = q["question_id"]
            m = models[i]
            a = model_answers[m].get(q_id)
            if a is None:
                print(f"Model {m} does not have answer for question {q_id}")
                continue
            if ref_answers is not None:
                ref = ref_answers['gpt-4'][q_id]
                matches.append(
                    MatchSingle(
                        dict(q), m, a, judge, ref_answer=ref, multi_turn=multi_turn
                    )
                )
            else:
                matches.append(MatchSingle(dict(q), m, a, judge, multi_turn=multi_turn))
    return matches


def make_judge_pairwise(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["pair-v2"])
    judges["math"] = Judge(judge_model, judge_prompts["pair-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["pair-v2-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["pair-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


def make_judge_single(judge_model, judge_prompts):
    judges = {}
    judges["default"] = Judge(judge_model, judge_prompts["single-v1"])
    judges["math"] = Judge(judge_model, judge_prompts["single-math-v1"], ref_based=True)
    judges["default-mt"] = Judge(
        judge_model, judge_prompts["single-v1-multi-turn"], multi_turn=True
    )
    judges["math-mt"] = Judge(
        judge_model,
        judge_prompts["single-math-v1-multi-turn"],
        ref_based=True,
        multi_turn=True,
    )
    return judges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--judge-file",
        type=str,
        default="llm_judge/data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )
    parser.add_argument("--judge-model-name", type=str, default="gpt-4", help="The model used for judging")
    parser.add_argument("--judge-model-url", type=str, default="", help="Base URL for the judge model API")
    parser.add_argument("--judge-model-api-key", type=str, default="", help="API key for the judge model")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )
    # Remove Azure parameter as we now use custom judge model parameters
    args = parser.parse_args()
    print(f"Model name: {args.model_list}")
    print(f"Judge model name: {args.judge_model_name}")
    if args.judge_model_url:
        print(f"Judge model URL: {args.judge_model_url}")
    if args.judge_model_api_key:
        print(f"Judge model API key: {args.judge_model_api_key[0:4]}***")

    args.model_list = [model_path.replace("/", "_") for model_path in args.model_list]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    question_file = os.path.join(current_dir, "data", args.bench_name, "question.jsonl")
    print(f"Loading questions from {question_file}...")
    answer_dir = os.path.join(current_dir, "data", args.bench_name, "model_answer")
    print(f"Loading answers from {answer_dir}...")
    ref_answer_dir = os.path.join(current_dir, "data", args.bench_name, "reference_answer")
    print(f"Loading reference answers from {ref_answer_dir}...")

    # Load questions
    questions = load_questions(str(question_file), None, None)

    # Load answers
    model_answers = load_model_answers(str(answer_dir))
    ref_answers = load_model_answers(str(ref_answer_dir))

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.mode == "single":
        judges = make_judge_single(args.judge_model_name, judge_prompts)
        play_a_match_func: Callable[[MatchSingle | MatchPair, str, dict[str, Any] | None], dict[str, Any]] = play_a_match_single
        model_suffix = "_".join(args.model_list)
        output_file = str(os.path.join(current_dir, "data", args.bench_name, "model_judgment", f"{args.judge_model_name}_{model_suffix}.jsonl"))
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model_name, judge_prompts)
        play_a_match_func: Callable[[MatchSingle | MatchPair, str, dict[str, Any] | None], dict[str, Any]] = play_a_match_pair
        output_file = str(os.path.join(current_dir, "data", args.bench_name, "model_judgment", f"{args.judge_model_name}_pair.jsonl"))
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model_name
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4, ensure_ascii=False))
    # input("Press Enter to confirm...")

    # Prepare API dict if judge model URL and API key are provided
    api_dict = None

    if args.judge_model_url or args.judge_model_api_key:
        api_dict = {}
        if args.judge_model_url:
            print(f"Using custom judge model URL: {args.judge_model_url}")
            api_dict["api_base"] = args.judge_model_url
        if args.judge_model_api_key:
            print(f"Using custom judge model API key: {args.judge_model_api_key[0:4]}***")
            api_dict["api_key"] = args.judge_model_api_key

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file, api_dict=api_dict)
    else:
        def play_a_match_wrapper(input_match):
            play_a_match_func(input_match, output_file=output_file, api_dict=api_dict)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass
