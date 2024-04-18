FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /opt/project

RUN  apt-get update && \
     apt-get install -y build-essential --no-install-recommends gcc git wget && \
     pip install --upgrade pip && \
     pip install numpy

COPY requirements.txt .

# RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt
RUN pip install -r requirements.txt

CMD ["bash"]