FROM python:3.9-slim

EXPOSE $PORT
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi --no-cache-dir
RUN pip install pydantic --no-cache-dir
RUN pip install uvicorn --no-cache-dir
RUN pip install google-cloud-storage --no-cache-dir
RUN pip install torch --no-cache-dir
RUN pip install torchvision --no-cache-dir
RUN pip install vit-pytorch --no-cache-dir
RUN pip install python-multipart
RUN touch image.jpg

COPY main.py main.py
COPY ../src/models/predict_model.py  ../src/models/predict_model.py
CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1
