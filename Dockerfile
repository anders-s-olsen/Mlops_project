# Base image
FROM python:3.8-slim
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
#COPY setup.py setup.py
COPY src/ src/
COPY models/ models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN mkdir data/
RUN mkdir data/raw
RUN mkdir data/processed
RUN python src/data/make_dataset.py
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
