#!/bin/bash

print_usage() {
    echo "Usage: $0 --model-name <model_name> [--openai-api-key <api_key> | (--judge-model-name <judge_model_name> --judge-model-url <url> --judge-model-api-key <api_key>)] --parallel <parallel>"
    echo
    echo "Arguments:"
    echo "  --model-name          Model name to be evaluated"
    echo "  --openai-api-key      OpenAI API key (backward compatibility)"
    echo "  --judge-model-name    Name of the judge model (default: gpt-4)"
    echo "  --judge-model-url     Base URL for the judge model API"
    echo "  --judge-model-api-key API key for the judge model"
    echo "  --parallel            Number of parallel processes"
    echo "  --ci                  CI mode"
}

OPENAI_API_KEY=""
MODEL_NAME=""
JUDGE_MODEL_NAME="gpt-4"
JUDGE_MODEL_URL=""
JUDGE_MODEL_API_KEY=""
PARALLEL="5"
CI="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --openai-api-key)
            OPENAI_API_KEY="$2"
            shift 2
            ;;
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

# If judge model API key is provided, use it
if [[ -n "$JUDGE_MODEL_API_KEY" ]]; then
    # Use the new judge model parameters
    if [[ -z "$JUDGE_MODEL_URL" ]]; then
        echo "Error: --judge-model-url is required when using --judge-model-api-key"
        print_usage
        exit 1
    fi
elif [[ -z "$OPENAI_API_KEY" ]]; then
    # Fall back to requiring OpenAI API key
    echo "Error: Either --judge-model-api-key or --openai-api-key is required"
    print_usage
    exit 1
else
    # If only OpenAI API key is provided, use it as the judge model API key
    JUDGE_MODEL_API_KEY="$OPENAI_API_KEY"
    # Default to OpenAI API URL if using OpenAI API key
    if [[ -z "$JUDGE_MODEL_URL" ]]; then
        JUDGE_MODEL_URL="https://api.openai.com/v1"
    fi
fi

if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model-name is required"
    print_usage
    exit 1
fi

export OPENAI_API_KEY="$OPENAI_API_KEY"
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
