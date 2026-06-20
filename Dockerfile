FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY models ./models
COPY src ./src
COPY service ./service
COPY weights ./weights
COPY tests ./tests
COPY DOTAv1.yaml .

RUN mkdir -p data/results

EXPOSE 8000 7860

CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]