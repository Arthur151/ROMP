#！/bin/bash
pip install -r requirements.txt

pip uninstall smplx
cd ../models/smplx
python setup.py install 

cd ../manopth
python setup.py install

cd ../../tools/
git clone https://github.com/liruilong940607/OCHumanApi
cd OCHumanApi
make install
git clone https://github.com/Jeff-sjtu/CrowdPose
cd CrowdPose/crowdpose-api/PythonAPI
sh install.sh
cd ../../

cd ..
# for centos 
#sudo yum install libXext libSM libXrender freeglut-devel

#for ubuntu: use pyrender in OSMesa mode
sudo apt update
sudo apt-get install libsm6 libxrender1 libfontconfig1 freeglut3-dev
sudo apt --fix-broken install
sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
sudo dpkg -i ./mesa_18.3.3-0.deb || true
sudo apt install -f

git clone https://github.com/mmatl/pyopengl.git
pip install ./pyopengl

cd ../src
