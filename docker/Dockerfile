FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America

RUN apt-get update && apt install -y software-properties-common; \
    add-apt-repository ppa:deadsnakes/ppa; \
    apt-get update; \
    apt install -y python3.9; \
    apt install -y python3-pip; \
    pip3 install torch>=1.13.1+cu117;

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
