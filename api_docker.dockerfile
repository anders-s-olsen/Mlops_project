FROM python:3.9-slim

EXPOSE $PORT
WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN touch image.jpg

COPY app.py app.py
COPY src/models/predict_model.py  src/models/predict_model.py
CMD exec uvicorn app:app --port $PORT --host 0.0.0.0 --workers 1
