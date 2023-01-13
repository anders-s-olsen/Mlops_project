# Base image
FROM  nvcr.io/nvidia/pytorch:22.07-py3
# install python
#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*
WORKDIR /
COPY requirements.txt requirements.txt
#COPY setup.py setup.py
COPY src/ src/
COPY models/ models/
RUN pwd
RUN pip install -r requirements.txt --no-cache-dir

RUN pwd
RUN mkdir data/
RUN mkdir data/raw
RUN mkdir data/processed
RUN python src/data/make_dataset.py

# Installs google cloud sdk, this is mostly for using gsutil to export model.
# RUN wget -nv \
#     https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
#     mkdir /tools && \
#     tar xvzf google-cloud-sdk.tar.gz -C /tools && \
#     rm google-cloud-sdk.tar.gz && \
#     /tools/google-cloud-sdk/install.sh --usage-reporting=false \
#         --path-update=false --bash-completion=false \
#         --disable-installation-options && \
#     rm -rf /.config/* && \
#     ln -s /.config /config && \
#     # Remove the backup directory that gcloud creates
#     rm -rf /tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
