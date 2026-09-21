FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# upload_dir/output_dir default to one level above the app package (see
# app/config.py) — /app/uploads and /app/outputs here, NOT /app/app/... .
# These paths must stay in step with the volume mounts in docker-compose.yml.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/uploads /app/outputs /app/data/demo \
    && chown -R appuser:appuser /app

ENV DEMO_DATA_DIR=/app/data/demo
ENV INFERENCE_MODE=stub

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
