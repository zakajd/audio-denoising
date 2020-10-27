FROM nvcr.io/nvidia/pytorch:20.06-py3

# base setup
RUN apt-get update --fix-missing && \
    apt-get install --no-install-recommends \
    libsm6 libxext6 libxrender-dev \
    wget software-properties-common pkg-config build-essential \
    libglu1-mesa -y && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/local/src/*

RUN pip --no-cache-dir install --upgrade \
    pipenv
    loguru

# adding this VAR to be able to rebuild last layer only by changing its value. Usefull for updating package inside docker
ARG RANDOM_VAR=0
RUN pip --no-cache-dir install --upgrade \
    git+https://github.com/bonlime/pytorch-tools.git@master

COPY . /workdir
WORKDIR /workdir