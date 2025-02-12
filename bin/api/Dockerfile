FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN pip install -e ".[model_worker,llm_judge]"

RUN mkdir -p llm_judge/data/japanese_mt_bench/model_judgment

ENV MODEL_NAME=""
ENV MODEL_API_KEY=""
ENV MODEL_URL=""
ENV PYTHONPATH=/app
ENV OPENAI_API_KEY=""

COPY bin/api/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
