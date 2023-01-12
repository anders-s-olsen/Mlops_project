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
COPY .dvc/ .dvc/
COPY data.dvc data.dvc
COPY .git .git
COPY .dvcignore .dvcignore
COPY .gitignore .gitignore

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN python src/data/make_dataset.py
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
