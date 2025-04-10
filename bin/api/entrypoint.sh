#!/bin/bash

# First argument determines the operation mode: generate or judge
MODE="$1"
shift

if [[ "$MODE" != "generate" && "$MODE" != "judge" ]]; then
    echo "Error: First argument must be either 'generate' or 'judge'"
    exit 1
fi

if [[ "$MODE" == "generate" ]]; then
    # Extract arguments for generate mode
    NUM_CHOICES="5"
    QUESTION_COUNT=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --num-choices)
                NUM_CHOICES="$2"
                shift 2
                ;;
            --question-count)
                QUESTION_COUNT="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    cmd=(python llm_judge/gen_api_answer_liquid.py \
        --bench-name japanese_mt_bench \
        --model "$MODEL_NAME" \
        --openai-api-key "$MODEL_API_KEY" \
        --openai-api-base "$MODEL_URL" \
        --num-choices "$NUM_CHOICES")

    if [[ -n "$QUESTION_COUNT" ]]; then
        cmd+=(--question-end "$QUESTION_COUNT")
    fi

    "${cmd[@]}"

elif [[ "$MODE" == "judge" ]]; then
    # Extract arguments for judge mode
    PARALLEL="5"
    CI="false"
    JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gpt-4"}
    JUDGE_MODEL_URL=${JUDGE_MODEL_URL:-""}
    JUDGE_MODEL_API_KEY=${JUDGE_MODEL_API_KEY:-""}
    
    # Ensure required parameters are set
    if [[ -z "$JUDGE_MODEL_API_KEY" ]]; then
        echo "Error: JUDGE_MODEL_API_KEY environment variable is required"
        exit 1
    fi
    
    if [[ -z "$JUDGE_MODEL_URL" ]]; then
        echo "Error: JUDGE_MODEL_URL environment variable is required"
        exit 1
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel)
                PARALLEL="$2"
                shift 2
                ;;
            --ci)
                CI="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done

    # Generate judgments
    python llm_judge/gen_judgment.py \
        --model-list "$MODEL_NAME" \
        --judge-model-name "$JUDGE_MODEL_NAME" \
        --judge-model-url "$JUDGE_MODEL_URL" \
        --judge-model-api-key "$JUDGE_MODEL_API_KEY" \
        --parallel "$PARALLEL" \
        --bench-name japanese_mt_bench

    # Show results
    python llm_judge/show_result.py \
        --model-list "$MODEL_NAME" \
        --judge-model-name "$JUDGE_MODEL_NAME" \
        --ci "$CI" \
        --bench-name japanese_mt_bench \
        --input-file "llm_judge/data/japanese_mt_bench/model_judgment/${JUDGE_MODEL_NAME}_$MODEL_NAME.jsonl" \
        --output "llm_judge/data/japanese_mt_bench/${JUDGE_MODEL_NAME}-score-$MODEL_NAME.json"
fi
