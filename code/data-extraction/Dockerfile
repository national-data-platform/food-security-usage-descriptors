FROM python:3.10.15-slim-bullseye

WORKDIR app

COPY *.py .
COPY src src
COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

EXPOSE 80
