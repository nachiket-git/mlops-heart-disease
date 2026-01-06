FROM python:3.11-slim

WORKDIR /app

# System deps (optional but common for sklearn wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
# (Optional) copy a pre-trained model into the image; in CI we train separately.
# COPY artifacts/model/final_pipeline.joblib /app/artifacts/model/final_pipeline.joblib

ENV MODEL_PATH=/app/artifacts/model/final_pipeline.joblib
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
