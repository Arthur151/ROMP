## Installation

#### Download models

###### Option 1:

Directly download the full-packed released package [ROMP.zip](https://github.com/Arthur151/ROMP/releases/download/v1.0/ROMP_v1.0.zip) from github, latest version v1.0.

###### Option 2:

Clone the repo:
```bash
git clone https://github.com/Arthur151/ROMP --depth 1
```

Then download the ROMP data from [Github release](https://github.com/Arthur151/ROMP/releases/download/v1.0/ROMP_data.zip), [Google drive](https://drive.google.com/file/d/1EZYEeLft5C2TkugaqsTP_wIsHVlWCyO8/view?usp=sharing). 

Unzip the downloaded ROMP_data.zip under the root ROMP/. 
```bash
cd ROMP/
unzip ROMP_data.zip
```

The layout would be
```bash
ROMP
  - demo
  - models
  - src
  - trained_models
```

#### Set up environments

Please intall the Pytorch 1.6 from [the official website](https://pytorch.org/). We have tested the code on Ubuntu 18.04 and Centos 7. 

Install packages:
```bash
cd ROMP/src
pip install -r requirements.txt
```