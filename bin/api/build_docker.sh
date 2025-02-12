#!/bin/bash

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Print help
print_usage() {
    echo "Usage: $0 [--no-cache]"
    echo
    echo "Options:"
    echo "  --no-cache  Build without using Docker cache"
}

# Process arguments
NO_CACHE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
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

# Build the Docker image using Dockerfile from bin/api
echo "Building Docker image liquidai/mt-bench:latest..."
docker build $NO_CACHE -t liquidai/mt-bench:latest -f bin/api/Dockerfile .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Build successful!"
else
    echo "Build failed!"
    exit 1
fi
