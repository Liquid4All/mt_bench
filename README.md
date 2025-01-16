# Run Evaluation through vLLM API

## Install

```bash
pip install -e ".[model_worker,llm_judge]"
```

## Run Evaluation

```bash
# run eval for lfm-3b-jp
PYTHONPATH=. python llm_judge/gen_api_answer_liquid.py \
  --bench-name japanese_mt_bench \
  --model lfm-3b-jp \
  --num-choices 5 \
  --openai-api-key <api-key> \
  --openai-api-base http://localhost:8000/v1

# run eval for lfm-3b-ichikara
PYTHONPATH=. python llm_judge/gen_api_answer_liquid.py \
  --bench-name japanese_mt_bench \
  --model lfm-3b-ichikara \
  --num-choices 5 \
  --openai-api-key <api-key> \
  --openai-api-base http://localhost:8000/v1
```
