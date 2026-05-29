FROM python:3.10-slim

WORKDIR /app

COPY scraper/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scraper/scraper.py .
COPY dags/scripts/corpus_cleaner.py .
COPY dags/scripts/universal_embedder.py .
COPY dags/scripts/generate_qa.py .
COPY dags/scripts/model_utils.py .

ENTRYPOINT ["python", "scraper.py"]
