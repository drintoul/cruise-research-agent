FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY app.py tools.py ingest.py chainlit.md ./
COPY .chainlit ./.chainlit
COPY public ./public

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "-h", "--host", "0.0.0.0", "--port", "8000"]
