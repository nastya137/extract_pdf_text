FROM python:3.11-slim

# Установка системных зависимостей для pdfplumber/PIL и curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Скачиваем swagger-ui статику в папку /app/static
RUN mkdir -p /app/static && \
    curl -sS https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js -o /app/static/swagger-ui-bundle.js && \
    curl -sS https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css -o /app/static/swagger-ui.css

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]