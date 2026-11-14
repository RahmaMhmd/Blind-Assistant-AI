FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AI_PORT=8000

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src ./src
COPY contracts ./contracts

RUN useradd --system --no-create-home service
USER service

EXPOSE ${AI_PORT}

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["python", "-m", "src.service.healthcheck"]

CMD ["python", "-m", "src.service"]
