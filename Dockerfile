FROM python:3.7.13-slim-buster

RUN apt-get update -y
RUN apt install gcc g++ -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install setuptools cython numpy
RUN pip install simple_romp

ENTRYPOINT [ "romp" ]
