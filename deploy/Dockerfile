FROM python:3.11-slim

WORKDIR /srv
COPY pyproject.toml ./
COPY app ./app
COPY backends ./backends
COPY config ./config
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" httpx pyyaml prometheus-client

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
