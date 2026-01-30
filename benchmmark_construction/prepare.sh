#!/bin/bash

if [ ! -f .env ]; then
    echo "Warning: .env file does not exist"
    echo "Please copy env.example to .env and set correct values:"
    echo "cp env.example .env"
    echo ""
    echo "Then set the following environment variables in .env file."
    exit 1
fi

set -a
source .env
set +a

echo "Environment variables loaded:"
echo "OPENAI_API_KEY: ${OPENAI_API_KEY}"
echo "OPENAI_BASE_URL: $OPENAI_BASE_URL"
echo "PYTHONPATH: $PYTHONPATH"
