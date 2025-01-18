# Run Evaluation through vLLM API

## Install

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

### Arguments

| Argument | Description | Value for on-prem stack | Required |
| --- | --- | --- | --- |
| `--model-name` | Model name | `lfm-3b-jp`, `lfm-3b-ichikara` | Yes |
| `--model-url` | Model URL | `http://localhost:8000/v1` | Yes |
| `--model-api-key` | API key for the model | `API_SECRET` in `.env` | Yes |
| `--num-choices` | Number of responses to generate for each question | `5` | No. Default to 1. |
| `--question-count` | Number of questions to run | None | No. Default to None, which runs all questions. |

Results will be output under `llm_judge/data/japanese_mt_bench/model_answer`. The filename has pattern `<model-name>.jsonl`.

## Run OpenAI Judgement

```bash
bin/api/run_openai_judge.sh --model-name lfm-3b-jp --openai-api-key <OPENAI-API-KEY>
bin/api/run_openai_judge.sh --model-name lfm-3b-ichikara --openai-api-key <OPENAI-API-KEY>
```

### Arguments

| Argument | Description | Required |
| --- | --- | --- |
| `--model-name` | Model name | Yes |
| `--openai-api-key` | OpenAI API key | Yes |
| `--parallel` | Number of parallel API calls | No. Default to 5. |

The results are output to `llm_judge/data/japanese_mt_bench/model_judgment/gpt-4_<model-name>.jsonl`.

## Acknowledgement

This repository is modified from [`FastChat/fastchat`](https://github.com/lm-sys/FastChat/tree/main/fastchat).
