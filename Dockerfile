FROM ubuntu:16.04
MAINTAINER "Arthur Tofani"

RUN apt-get update && apt-get install -y wget ca-certificates \
    build-essential wget cmake pkg-config \
    libgtk-3-dev \
    git curl vim nano python3-dev python3-pip \
    libhdf5-dev
#rm -rf /var/lib/apt/lists/*

RUN apt-get install -y --no-install-recommends libedit-dev build-essential

RUN apt-get install -y apt-utils
RUN apt-get install -y ffmpeg
RUN apt-get install -y --no-install-recommends  llvm-8 llvm-8-dev

WORKDIR /home


RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas sklearn matplotlib seaborn jupyter pyyaml h5py
RUN LLVM_CONFIG=/usr/bin/llvm-config-8 pip3 install enum34 llvmlite numba
RUN pip3 install librosa==0.7.2
RUN pip3 install deepdish psutil
RUN pip3 install cython
RUN pip3 install datasketch
RUN pip3 install madmom essentia
RUN pip3 install matrixprofile-ts
RUN pip3 install git+https://github.com/bmcfee/crema.git@master
RUN pip3 install hashedindex
RUN pip3 install elasticsearch
RUN pip3 install scipy crema
