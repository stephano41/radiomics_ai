#FROM pytorch/pytorch:latest
#RUN pip install --no-cache-dir --user -r /requirements.txt

# final stage
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential --no-install-recommends gcc && \
    apt-get install -y git

COPY requirements.txt .

RUN pip install numpy
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

#COPY --from=builder /root/.local/lib/python3.10/site-packages /usr/local/lib/python3.10/dist-packages

COPY . .

