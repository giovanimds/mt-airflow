FROM docker.io/apache/airflow:3.2.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Install additional python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
