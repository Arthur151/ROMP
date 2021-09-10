#!/usr/bin/env bash

export CONDA_ENV_NAME=3dpw_eval
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

conda install numpy
pip install opencv-python
pip install numpy-quaternion
pip install scipy
pip install chumpy
# 4 pip install
# 5 pip install scipy

wget https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip
unzip sequenceFiles.zip
rm sequenceFiles.zip
mkdir input_dir
mkdir ./input_dir/ref
mv ./sequenceFiles/test ./sequenceFiles/train ./sequenceFiles/validation ./input_dir/ref
