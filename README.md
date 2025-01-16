# Run Evaluation through vLLM API

## Install

```bash
bin/api/prepare.sh
```

## Run Evaluation

```bash
# run eval for lfm-3b-jp
bin/api/run_api_eval.sh \
  --model-name lfm-3b-jp \
  --model-url http://localhost:8000/v1 \
  --model-api-key <API-KEY>

# run eval for lfm-3b-ichikara
bin/api/run_api_eval.sh \
  --model-name lfm-3b-ichikara \
  --model-url http://localhost:8000/v1 \
  --model-api-key <API-KEY>
```

## Evaluation results

Results will be output under `llm_judge/data/japanese_mt_bench/model_answer`. The filename has pattern `<model-name>-<timestamp>.jsonl`.
