#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

print_usage() {
    echo "Usage: $0 <generate|judge> [options]"
    echo
    echo "Commands:"
    echo "  generate    Generate model answers"
    echo "  judge      Run GPT-4 judgments"
    echo
    echo "Generate mode options:"
    echo "  --model-name      Name of the model to evaluate"
    echo "  --model-api-key   API key for model access (optional)"
    echo "  --model-url       Base URL for the model API"
    echo "  --num-choices     Number of choices to generate (default: 5)"
    echo "  --question-count  Number of questions to evaluate (optional)"
    echo
    echo "Judge mode options:"
    echo "  --model-name      Name of the model to evaluate"
    echo "  --openai-api-key  OpenAI API key for GPT-4 judgment"
    echo "  --parallel        Number of parallel processes (default: 5)"
    echo "  --ci              CI mode (default: false)"
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

MODE="$1"
shift

if [[ "$MODE" != "generate" && "$MODE" != "judge" ]]; then
    echo "Error: First argument must be either 'generate' or 'judge'"
    print_usage
    exit 1
fi

if [[ "$MODE" == "generate" ]]; then
    # Process generate mode arguments
    MODEL_NAME=""
    MODEL_API_KEY="placeholder"
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

    if [[ -z "$MODEL_URL" ]]; then
        echo "Error: --model-url is required"
        print_usage
        exit 1
    fi

    # Run generate mode
    docker run \
        -e MODEL_NAME="$MODEL_NAME" \
        -e MODEL_API_KEY="$MODEL_API_KEY" \
        -e MODEL_URL="$MODEL_URL" \
        -v "$(pwd)/llm_judge:/app/llm_judge" \
        liquidai/mt-bench:latest generate \
        --num-choices "$NUM_CHOICES" \
        --question-count "$QUESTION_COUNT"

elif [[ "$MODE" == "judge" ]]; then
    # Process judge mode arguments
    MODEL_NAME=""
    OPENAI_API_KEY=""
    PARALLEL="5"
    CI="false"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --openai-api-key)
                OPENAI_API_KEY="$2"
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

    if [[ -z "$MODEL_NAME" ]]; then
        echo "Error: --model-name is required"
        print_usage
        exit 1
    fi

    if [[ -z "$OPENAI_API_KEY" ]]; then
        echo "Error: --openai-api-key is required"
        print_usage
        exit 1
    fi

    # Run judge mode
    docker run --rm -it \
        --network="host" \
        -e MODEL_NAME="$MODEL_NAME" \
        -e OPENAI_API_KEY="$OPENAI_API_KEY" \
        -v "$(pwd)/llm_judge:/app/llm_judge" \
        liquidai/mt-bench:latest judge \
        --parallel "$PARALLEL" \
        --ci "$CI"
fi
