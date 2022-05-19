FROM python:3.7.13-slim-buster

RUN apt-get update -y
RUN apt install gcc g++ -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install setuptools cython numpy
RUN pip install simple_romp

ENTRYPOINT [ "romp" ]

# romp --mode=image --input 13pic1.jpg -o output --render_mesh

# docker build -t romp . 
# docker run --privileged --rm -it --gpus 0 --ipc=host -p 8888:8888 -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 -e DISPLAY=$DISPLAY romp --mode=webcam