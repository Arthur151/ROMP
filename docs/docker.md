## Docker usage

```
# Build
docker build --rm -t romp .
# Inference
docker run --privileged --rm -it --gpus 0 --ipc=host -p 8888:8888 -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/workspace/results --device /dev/video0 -e DISPLAY=$DISPLAY romp --mode=webcam
```