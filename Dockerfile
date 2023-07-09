FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y build-essential --no-install-recommends gcc && \
    apt-get install -y git && \
    pip install --upgrade pip && \
    pip install numpy

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
#RUN pip install -r requirements.txt
RUN pip install --no-deps shap==0.42

#COPY . .

#CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
#EXPOSE 8000
CMD ["mlflow", "ui", "--host=0.0.0.0", "--port=5000", "--backend-store-uri=./outputs/mlflow"]
EXPOSE 5000