#!/bin/bash

print_usage() {
    echo "Usage: $0 --model-name <model_name> --model-api-key <api_key> --model-url <base_url> --num-choices <num-choices> --question-count <question_count>"
    echo
    echo "Arguments:"
    echo "  --model-name     Name of the model to evaluate"
    echo "  --model-api-key  API key for model access"
    echo "  --model-url Base URL for the model API"
    echo "  --num-choices    Number of choices to generate for each question (default to 5)"
    echo "  --question-count Number of questions to evaluate (default to none, which runs all questions)"
}

MODEL_NAME=""
MODEL_API_KEY=""
MODEL_URL=""
NUM_CHOICES="5"
QUESTION_COUNT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --model-api-key)
            MODEL_API_KEY="$2"
            shift 2
            ;;
        --model-url)
            MODEL_URL="$2"
            shift 2
            ;;
        --num-choices)
            NUM_CHOICES="$2"
            shift 2
            ;;
        --question-count)
            QUESTION_COUNT="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_NAME" ]]; then
    echo "Error: --model-name is required"
    print_usage
    exit 1
fi

if [[ -z "$MODEL_API_KEY" ]]; then
    echo "Error: --model-api-key is required"
    print_usage
    exit 1
fi

if [[ -z "$MODEL_URL" ]]; then
    echo "Error: --model-url is required"
    print_usage
    exit 1
fi

export PYTHONPATH=.

python llm_judge/gen_api_answer_liquid.py \
  --bench-name japanese_mt_bench \
  --model "$MODEL_NAME" \
  --openai-api-key "$MODEL_API_KEY" \
  --openai-api-base "$MODEL_URL" \
  --num-choices "$NUM_CHOICES" \
  --question-end "$QUESTION_COUNT"
