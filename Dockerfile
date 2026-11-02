FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-serving.txt .
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements-serving.txt
ENV PYTHONPATH=/install/lib/python3.11/site-packages

COPY src/pipeline/ src/pipeline/
COPY src/train_final_model.py src/train_final_model.py

RUN mkdir -p data/raw data/processed models \
    && python src/pipeline/collect.py --output data/raw/iris_raw.csv \
    && python src/pipeline/preprocess.py \
        --input data/raw/iris_raw.csv \
        --output data/processed/iris_preprocessed.csv \
    && python src/pipeline/features.py \
        --input data/processed/iris_preprocessed.csv \
        --output data/processed/iris_features.csv \
    && python src/pipeline/validate.py \
        --input data/processed/iris_features.csv \
    && python src/train_final_model.py

FROM python:3.11-slim AS runtime

ENV PYTHONPATH=/install/lib/python3.11/site-packages \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY --from=builder /install/ /install/
COPY --from=builder --chown=app:app /build/models/ /app/models/
COPY --chown=app:app src/serve.py /app/src/serve.py

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["python", "src/serve.py"]
