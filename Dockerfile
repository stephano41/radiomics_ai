FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential --no-install-recommends gcc && \
    apt-get install -y git && \
    pip install --upgrade pip && \
    pip install numpy

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

COPY . .

