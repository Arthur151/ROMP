## Basic Installation

To run the demo code, please install via
```
pip install simple-romp
```

### (Optional) Install Pytorch GPU

If you have a GPU on your computer, for real-time inference, we recommand to install [Pytorch](https://pytorch.org/) GPU version. 

Here we give an example to install pytorch 1.10.0. 

#### 1. Install [Pytorch](https://pytorch.org/).
Please choose one of the following 4 options to install Pytorch via [conda](https://docs.conda.io/en/latest/miniconda.html) or [pip](https://pip.pypa.io/en/stable). 
Here, we support to install with Python 3.9, 3.8 or 3.7. 
We recommend installing via conda (Option 1-3) so that ROMP env is clean and will not affect other repo.  

##### Option 1) to install conda env with python 3.9, please run
```
conda create -n ROMP python=3.9
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```
##### Option 2) to install conda env with python 3.8, please run
```
conda create -n ROMP python==3.8.8  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```

##### Option 3) to install conda env with python 3.7, please run
```
conda create -n ROMP python==3.7.6  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```

##### Option 4) install via pip
To directly install via pip, you need to install CUDA 10.2 first (For Ubuntu, run`sudo apt-get install cuda-10-2`).  
Then install pytorch via:  
```
pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```