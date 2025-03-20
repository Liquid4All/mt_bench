#!/bin/bash

print_usage() {
    echo "Usage: $0 --model-name <model_name> --judge-model-name <judge_model_name> --judge-model-url <url> --judge-model-api-key <api_key> --parallel <parallel>"
    echo
    echo "Arguments:"
    echo "  --model-name          Model name to be evaluated"
    echo "  --judge-model-name    Name of the judge model (default: gpt-4)"
    echo "  --judge-model-url     Base URL for the judge model API"
    echo "  --judge-model-api-key API key for the judge model"
    echo "  --parallel            Number of parallel processes"
    echo "  --ci                  CI mode"
}

MODEL_NAME=""
JUDGE_MODEL_NAME="gpt-4"
JUDGE_MODEL_URL=""
JUDGE_MODEL_API_KEY=""
PARALLEL="5"
CI="false"

while [[ $# -gt 0 ]]; do
    case $1 in

        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --judge-model-name)
            JUDGE_MODEL_NAME="$2"
            shift 2
            ;;
        --judge-model-url)
            JUDGE_MODEL_URL="$2"
            shift 2
            ;;
        --judge-model-api-key)
            JUDGE_MODEL_API_KEY="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --ci)
            CI="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$JUDGE_MODEL_API_KEY" ]]; then
    echo "Error: --judge-model-api-key is required"
    print_usage
    exit 1
fi

if [[ -z "$JUDGE_MODEL_URL" ]]; then
    echo "Error: --judge-model-url is required"
    print_usage
    exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model-name is required"
    print_usage
    exit 1
fi

export JUDGE_MODEL_NAME="$JUDGE_MODEL_NAME"
export JUDGE_MODEL_URL="$JUDGE_MODEL_URL"
export JUDGE_MODEL_API_KEY="$JUDGE_MODEL_API_KEY"
export PYTHONPATH=.

python llm_judge/gen_judgment.py \
  --model-list "$MODEL_NAME" \
  --judge-model "$JUDGE_MODEL_NAME" \
  --parallel "$PARALLEL" \
  --bench-name japanese_mt_bench

python llm_judge/show_result.py --model-list "$MODEL_NAME" \
  --judge-model "$JUDGE_MODEL_NAME" \
  --ci "$CI" \
  --bench-name japanese_mt_bench \
  --input-file "llm_judge/data/japanese_mt_bench/model_judgment/${JUDGE_MODEL_NAME}_$MODEL_NAME.jsonl" \
  --output "llm_judge/data/japanese_mt_bench/${JUDGE_MODEL_NAME}-score-$MODEL_NAME.json"
