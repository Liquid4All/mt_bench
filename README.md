# Run Evaluation through vLLM API

## Overview

1. Run the model through vLLM with an OpenAI compatible API.
  - For Liquid models, run the on-prem stack, or use Liquid [`labs`](https://labs.liquid.ai).
  - For other models, use the `run-vllm.sh` script, or use 3rd party providers.
2. Run the evaluation script with the model API endpoint and API key.
  - The evaluation can be run with Docker (recommended) or locally without Docker.

## Run Evaluation with Docker

1. Generate model answers:

```bash
bin/api/run_docker_eval.sh generate \
  --model-name <model-name> \
  --model-url <model-url> \
  --model-api-key <model-api-key>
```

Results will be output in `llm_judge/data/japanese_mt_bench/model_answer/<model-name>.jsonl`

2. Run judge:

```bash
bin/api/run_docker_eval.sh judge \
  --model-name <model-name> \
  --judge-model-name <judge-model-name> \
  --judge-model-url <judge-model-url> \
  --judge-model-api-key <judge-model-api-key>
```

Judge results will be output to `llm_judge/data/japanese_mt_bench/model_judgment/<judge-model-name>_<model-name>.jsonl`.

The final scores will be output in `llm_judge/data/japanese_mt_bench/<judge-model-name>-score-<model-name>.json`.

### Examples

Run evaluation for `lfm-3b-jp` on-prem:

```bash
bin/api/run_docker_eval.sh generate \
  --model-name lfm-3b-jp \
  --model-url http://localhost:8000/v1 \
  --model-api-key <ON-PREM-API-SECRET>

bin/api/run_docker_eval.sh judge \
  --model-name lfm-3b-jp \
  --judge-model-name gpt-4o \
  --judge-model-url https://api.openai.com/v1 \
  --judge-model-api-key <OPENAI-API-KEY>
```

Run eval for `lfm-3b-ichikara` on-prem:

```bash
bin/api/run_docker_eval.sh generate \
  --model-name lfm-3b-ichikara \
  --model-url http://localhost:8000/v1 \
  --model-api-key <ON-PREM-API-SECRET>

bin/api/run_docker_eval.sh judge \
  --model-name lfm-3b-ichikara \
  --openai-api-key <OPENAI-API-KEY>
```

Run eval for `lfm-3b-jp` on `labs`:

```bash
bin/api/run_docker_eval.sh generate \
  --model-name lfm-3b-jp \
  --model-url https://inference-1.liquid.ai/v1 \
  --model-api-key <API-KEY>

bin/api/run_docker_eval.sh judge \
  --model-name lfm-3b-jp \
  --judge-model-name gpt-4o \
  --judge-model-url https://api.openai.com/v1 \
  --judge-model-api-key <OPENAI-API-KEY>
```

## Run Evaluation without Docker

<details>

<summary>(click to see details)</summary>

### Install

It is recommended to create a brand new `conda` environment first. But this step is optional.

```bash
conda create -n mt_bench python=3.10
conda activate mt_bench
```

Run the following command to set up the environment and install the dependencies:

```bash
bin/api/prepare.sh
```

### Run Evaluation

1. Run `bin/api/run_api_eval.sh` script to generate model answers.

```bash
bin/api/run_api_eval.sh \
  --model-name <model-name> \
  --model-url <model-url> \
  --model-api-key <API-KEY>
```

Results will be output in `llm_judge/data/japanese_mt_bench/model_answer/<model-name>.jsonl`.

2. Run the following scripts to generate GPT-4 judgement scores for the model answers.

```bash
bin/api/run_openai_judge.sh --model-name <model-name> --judge-model-name <judge-model-name> --judge-model-url <judge-model-url> --judge-model-api-key <judge-model-api-key>

# examples:
bin/api/run_openai_judge.sh --model-name lfm-3b-jp --judge-model-name gpt-4o --judge-model-url https://api.openai.com/v1 --judge-model-api-key <OPENAI-API-KEY>
bin/api/run_openai_judge.sh --model-name lfm-3b-ichikara --judge-model-name gpt-4o --judge-model-url https://api.openai.com/v1 --judge-model-api-key <OPENAI-API-KEY>
```

Judge results will be output to `llm_judge/data/japanese_mt_bench/model_judgment/<judge-model-name>_<model-name>.jsonl`.

The final scores will be output in `llm_judge/data/japanese_mt_bench/<judge-model-name>-score-<model-name>.json`.

</details>

## Script Parameters

<details>

<summary>(click to see details)</summary>

### Generate Script Params

This applies to both `bin/api/run_docker_eval.sh generate` and `bin/api/run_api_eval.sh`.

| Argument | Description | Value for on-prem stack | Required |
| --- | --- | --- | --- |
| `--model-name` | Model name | `lfm-3b-jp`, `lfm-3b-ichikara` | Yes |
| `--model-url` | Model URL | `http://localhost:8000/v1` | Yes |
| `--model-api-key` | API key for the model | `API_SECRET` in `.env` | Yes |
| `--num-choices` | Number of responses to generate for each question | `5` | No. Default to 5. |
| `--question-count` | Number of questions to run | None | No. Default to None, which runs all questions. |

### Judge Script Params

This applies to both `bin/api/run_docker_eval.sh judge` and `bin/api/run_openai_judge.sh`.

| Argument | Description | Required |
| --- | --- | --- |
| `--model-name` | Model name to be evaluated | Yes |
| `--judge-model-name` | Name of the judge model (default: gpt-4) | No |
| `--judge-model-url` | Base URL for the judge model API | Yes |
| `--judge-model-api-key` | API key for the judge model | Yes |
| `--parallel` | Number of parallel API calls | No. Default to 5. |

</details>

## Acknowledgement

This repository is modified from [`FastChat/fastchat`](https://github.com/lm-sys/FastChat/tree/main/fastchat).
