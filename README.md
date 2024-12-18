# LLM Judge
| [Paper](https://arxiv.org/abs/2306.05685) | [Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) |

In this package, you can use MT-bench questions and prompts to evaluate your models with LLM-as-a-judge.
MT-bench is a set of challenging multi-turn open-ended questions for evaluating chat assistants.
To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.

## Contents
- [Install](#install)
- [Review Pre-Generated Model Answers and Judgments](#review-pre-generated-model-answers-and-judgments)
- [MT-Bench](#mt-bench)
- [Agreement Computation](#agreement-computation)
- [Datasets](#datasets)
- [Citation](#citation)

## Install
Install the lfm repo to run liquid hf models
https://github.com/Liquid4All/liquid_lfm

after lfm is installed install the following:
```
pip install -e ".[model_worker,llm_judge]"
pip install openai==0.28.1
```

## MT-Bench

### Evaluate a model on MT-bench

#### Step 1. Generate model answers to MT-bench questions
Code to run JP MT Bench for LFMs

```
cd llm_judge
srun --gpus=1 python gen_model_answer_liquid.py --model-path /lambdafs/checkpoints/maxime_3B_sft298860_dpo_dpoliquid_epoch2_302062_HF --model-id lfm-3b-jp-hf --bench-name japanese_mt_bench --num-choices 5
```

Code to run JP MT Bench via labs API
```
cd llm_judge
python3 gen_api_answer_liquid.py --bench-name japanese_mt_bench --model lfm-3b-jp --openai-api-key <labs-api-key> --openai-api-base https://inference-1.liquid.ai/v1 --num-choices 5
```

Note, --num choices should be set to 5 in the Swallow evals.

To run open-source models
```
cd llm_judge
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
```

The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl`.


#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
cd llm_judge
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call] --bench-name japanese_mt_bench
```

e.g.,
```
cd llm_judge
python gen_judgment.py --model-list lfm-3b-jp --parallel 2 --bench-name japanese_mt_bench
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench scores

- Show the scores for selected models
  ```
  cd llm_judge
  python show_result.py --model-list lfm-3b-jp --bench-name japanese_mt_bench --output-file <output_location.json> --input-file <input_location.json>
  ```
---

### How to get GPT-3.5/GPT-4/Claude's answer?
- `python gen_api_answer.py --model [MODEL-NAME]` to generate GPT-3.5/4 and Claude's answers.


### Other backends
We can also use vLLM for answer generation, which can be faster for the models supported by vLLM.

1. Launch a vLLM worker
```
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.vllm_worker --model-path [MODEL-PATH]
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```
  - Arguments:
    - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.

2. Generate the answers
```
python gen_api_answer.py --model [MODEL-NAME] --openai-api-base http://localhost:8000/v1 --parallel 50
```
  - Arguments:
    - `[MODEL-NAME]` is the name of the model from Step 1.
    - `--parallel` is the number of concurrent API calls to the vLLM worker.


## Datasets
- [Chatbot Arena Conversation Dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
- [MT-bench Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
