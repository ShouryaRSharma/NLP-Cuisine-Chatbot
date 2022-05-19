FROM nvidia/cuda:11.6.0-base-ubuntu18.04
CMD nvidia-smi

FROM python:3.9

WORKDIR /bot

COPY requirements.txt .
COPY environment.yml .
COPY shell.sh .

RUN apt-get update
RUN apt-get install -y wget

RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

COPY ./application ./application

CMD shell.sh

FROM python:3.9
CMD ["python", "./application/app.py"]


