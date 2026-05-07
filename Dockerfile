FROM python:3.11-slim

# Системные зависимости: libjpeg для pdfplumber, curl для Swagger, unzip для словаря
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Статика для локального Swagger
RUN mkdir -p /app/static && \
    curl -sS https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js -o /app/static/swagger-ui-bundle.js && \
    curl -sS https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css -o /app/static/swagger-ui.css

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ЗАРАНЕЕ скачанный словарь и распаковываем
COPY words.zip /tmp/words.zip
RUN mkdir -p /root/nltk_data/corpora && \
    unzip -o /tmp/words.zip -d /root/nltk_data/corpora && \
    rm /tmp/words.zip

COPY ./app ./app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]