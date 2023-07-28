FROM python:3.10-slim

WORKDIR /opt/project

RUN apt-get update && \
    apt-get install -y build-essential --no-install-recommends gcc && \
    apt-get install -y git && \
    pip install --upgrade pip && \
    pip install numpy

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
#RUN pip install -r requirements.txt
RUN pip install --no-deps shap==0.42 pandas==1.4.3

#COPY . .

CMD ["mlflow", "ui", "--host=0.0.0.0", "--port=8000", "--backend-store-uri=./outputs/models"]
EXPOSE 8000