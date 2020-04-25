FROM amaksimov/python_data_science
RUN apt-get update
RUN apt-get install -y ffmpeg nano
RUN apt-get install -y git

WORKDIR /home

RUN pip3 install --upgrade pip
RUN pip3 install wget gdown
RUN pip3 install librosa==0.7.2
RUN pip3 install numba deepdish psutil
RUN pip3 install cython
RUN pip3 install madmom essentia
RUN pip3 install matrixprofile-ts
RUN pip3 install git+https://github.com/bmcfee/crema.git@master
RUN pip3 install hashedindex
RUN pip3 install elasticsearch

RUN git clone https://github.com/arthurtofani/footprint.git footprint-repo
RUN ln -s footprint-repo footprint

#CMD ["jupyter", "lab", "--allow-root"]
