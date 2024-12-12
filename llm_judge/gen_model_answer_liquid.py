"""Generate answers with local models.

Usage:
python3 gen_model_answer_liquid.py --model-path /path/to/model --model-id my_model_id [--debug]
"""

import argparse
import json
import os
import random
import time
import importlib.util

import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import get_conversation_template
from fastchat.utils import str_to_torch_dtype, set_seed


def debug_print(msg: str, debug: bool):
    """Print debug messages only if debug is True."""
    if debug:
        print(f"[DEBUG] {msg}")


def load_model(
    model_path: str,
    revision: str = "main",
    device: str = "cuda",
    num_gpus: int = 1,
    max_gpu_memory: str = None,
    dtype: str = None,  # Allow dtype to be None
    load_8bit: bool = False,
    cpu_offloading: bool = False,
    debug: bool = False,
):
    """
    Load a Hugging Face model from a local path with support for multiple GPUs, data types,
    8-bit loading, and CPU offloading.
    """
    debug_print(f"Loading model from {model_path}", debug)

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        torch_dtype = dtype_mapping.get(dtype.lower(), torch.float32)

    device_map = "auto" if device == "cuda" else None
    if max_gpu_memory:
        device_map = {f"cuda:{i}": max_gpu_memory for i in range(num_gpus)} if num_gpus > 1 else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_8bit=load_8bit,
        offload_folder="offload" if cpu_offloading else None,
        offload_state_dict=cpu_offloading,
        attn_implementation="flash_attention_2"
    )

    if device == "cuda" and num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))

    if debug:
        debug_print("Model and tokenizer loaded successfully.", debug)
        debug_print(f"Model class: {model.__class__.__name__}", debug)
        debug_print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}", debug)

    return model, tokenizer


def load_module(path: str):
    if not path:
        raise ValueError(f"Can't find model file in {path}")
    module_name = os.path.basename(path).rsplit(".", 1)[0]
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None:
        raise ImportError(f"Can't find module {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def register_custom_model(model_path: str, debug: bool):
    model_file = os.path.join(model_path, 'modeling_liquid_hybrid.py')
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Modeling file not found: {model_file}")

    module = load_module(model_file)

    config_class_name = "LiquidHybridConfig"
    model_class_name = "LiquidHybridForCausalLM"

    if not hasattr(module, config_class_name):
        raise ImportError(f"Custom config class '{config_class_name}' not found.")
    if not hasattr(module, model_class_name):
        raise ImportError(f"Custom model class '{model_class_name}' not found.")

    custom_config_class = getattr(module, config_class_name)
    custom_model_class = getattr(module, model_class_name)

    AutoConfig.register("liquid_hybrid", custom_config_class)
    AutoModel.register(custom_config_class, custom_model_class)
    AutoModelForCausalLM.register(custom_config_class, custom_model_class)

    debug_print(f"Registered custom model architecture 'liquid_hybrid' from {model_path}", debug)


def is_local_path(path):
    return os.path.exists(path)


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    debug
):
    questions = load_questions(question_file, question_begin, question_end)
    #print(f"Loaded {len(questions)} questions.")
    random.shuffle(questions)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                debug=debug,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    debug=False,
):
    if is_local_path(model_path):
        register_custom_model(model_path, debug)
    else:
        debug_print("Using a remote model from Hugging Face. No custom model registration.", debug)

    model, tokenizer = load_model(
        model_path=model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=debug,
    )

    for question in tqdm(questions):
        temperature = 0.7
        top_p = 1
        top_k = -1
        min_p = 0.1 
        repetition_penalty = 1.05
        max_new_token = 512
        seed = 14
        choices = []

        for i in range(num_choices):
            torch.manual_seed(seed)  # Setting the seed for reproducibility
            conv = get_conversation_template(model_id)
            debug_print(f"Processing question {question['question_id']} with seed {seed}", debug)
            turns = []
            prompts = []

            for j, qs in enumerate(question["turns"]):
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)

                do_sample = True if temperature >= 1e-4 else False

                try:
                    input_ids = tokenizer([prompt]).input_ids
                    input_ids = torch.as_tensor(input_ids).cuda()
                    print("input shape", input_ids.shape)
                    start_time = time.time()
                    output_ids = model.generate(
                        input_ids,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        top_k=top_k if top_k >= 0 else None,  # Use `None` if top_k is -1
                        repetition_penalty=repetition_penalty,
                        max_new_tokens=max_new_token,
                    )
                    print("Generate time:", time.time() - start_time)

                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [idx for idx, t_id in enumerate(output_ids) if t_id in conv.stop_token_ids]
                        if stop_token_ids_index:
                            output_ids = output_ids[:stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )

                    if conv.stop_str:
                        if isinstance(conv.stop_str, list):
                            stop_str_indices = [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                            if stop_str_indices:
                                output = output[: min(stop_str_indices)]
                        else:
                            idx = output.find(conv.stop_str)
                            if idx > 0:
                                output = output[:idx]

                    # Clean special tokens
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for st in special_token:
                                output = output.replace(st, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()

                except RuntimeError as e:
                    print(f"ERROR question ID: {question['question_id']}, Error: {str(e)}")
                    output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a", encoding='utf-8') as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

    debug_print("Finished generating all answers.", debug)


def reorg_answer_file(answer_file):
    """Sort by question id and remove duplicates."""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maximum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=14,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug printing."
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()

    args.model_id = args.model_id.replace("/", "_")

    question_file = f"/fastchat/llm_judge/data/japanese_mt_bench/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")
    print(f"Set random seed to {args.seed}")
    set_seed(args.seed)

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        debug=args.debug,
    )

    reorg_answer_file(answer_file)