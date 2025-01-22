# Run Evaluation through vLLM API

## Install

It is recommended to create a brand new `conda` environment first. But this step is optional.

```bash
conda create -n mt_bench python=3.10
conda activate mt_bench
```

Run the following command to set up the environment and install the dependencies:

```bash
bin/api/prepare.sh
```

## Run Evaluation

To run the evaluation locally, first launch the on-prem stack following the instruction.

Run the following commands to launch the evaluation:

```bash
# run eval for lfm-3b-jp
bin/api/run_api_eval.sh \
  --model-name lfm-3b-jp \
  --model-url http://localhost:8000/v1 \
  --model-api-key <API-KEY> \
  --num-choices 5

# run eval for lfm-3b-ichikara
bin/api/run_api_eval.sh \
  --model-name lfm-3b-ichikara \
  --model-url http://localhost:8000/v1 \
  --model-api-key <API-KEY> \
  --num-choices 5
```

<details>

<summary>(click to see more details about the evaluation script)</summary>

### Arguments

| Argument | Description | Value for on-prem stack | Required |
| --- | --- | --- | --- |
| `--model-name` | Model name | `lfm-3b-jp`, `lfm-3b-ichikara` | Yes |
| `--model-url` | Model URL | `http://localhost:8000/v1` | Yes |
| `--model-api-key` | API key for the model | `API_SECRET` in `.env` | Yes |
| `--num-choices` | Number of responses to generate for each question | `5` | No. Default to 1. |
| `--question-count` | Number of questions to run | None | No. Default to None, which runs all questions. |

Results will be output under `llm_judge/data/japanese_mt_bench/model_answer`. The filename has pattern `<model-name>.jsonl`.

</details>

## Get OpenAI Judgement Scores

The following scripts will generate GPT-4 judgement scores for the models.

```bash
bin/api/run_openai_judge.sh --model-name lfm-3b-jp --openai-api-key <OPENAI-API-KEY>
bin/api/run_openai_judge.sh --model-name lfm-3b-ichikara --openai-api-key <OPENAI-API-KEY>
```

GPT judge results will be output under `llm_judge/data/japanese_mt_bench/model_judgment`. The filename has pattern `gpt-4_<model-name>.jsonl`.

The final scores will be output in `data/japanese_mt_bench/gpt4-score-<model-name>.json`.

<details>

<summary>(click to see more details about the evaluation script)</summary>

### Arguments

| Argument | Description | Required |
| --- | --- | --- |
| `--model-name` | Model name | Yes |
| `--openai-api-key` | OpenAI API key | Yes |
| `--parallel` | Number of parallel API calls | No. Default to 5. |

The results are output to `llm_judge/data/japanese_mt_bench/model_judgment/gpt-4_<model-name>.jsonl`.

</details>

## Acknowledgement

This repository is modified from [`FastChat/fastchat`](https://github.com/lm-sys/FastChat/tree/main/fastchat).
