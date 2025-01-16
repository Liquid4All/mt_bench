#!/bin/bash

print_usage() {
    echo "Usage: $0 --openai-api-key <api_key> --model-name <model_name> --parallel <parallel>"
    echo
    echo "Arguments:"
    echo "  --openai-api-key OpenAI API key"
    echo "  --model-name     Model name"
    echo "  --parallel       Number of parallel processes"
}

OPENAI_API_KEY=""
MODEL_NAME=""
PARALLEL="5"

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
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "OPENAI_API_KEY" ]]; then
    echo "Error: --openai-api-key is required"
    print_usage
    exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model-name is required"
    print_usage
    exit 1
fi

export OPENAI_API_KEY="$OPENAI_API_KEY"
export PYTHONPATH=.

python llm_judge/gen_judgment.py \
  --model-list "$MODEL_NAME" \
  --parallel "$PARALLEL" \
  --bench-name japanese_mt_bench
