## Bugs

### libstdc++.so.6: version 'GLIBCXX_3.4.21' not found  
``bash
sudo find / -name libstdc++.so.6*  
# find the correct lib in the search results
such as, /export/home/suny/anaconda3/lib/libstdc++.so.6.0.26 in my system
# set the link to this lib
sudo rm -rf /usr/lib64/libstdc++.so.6  
sudo ln -s /export/home/suny/anaconda3/lib/libstdc++.so.6.0.26 /usr/lib64/libstdc++.so.6  
``

### Visualization bug: OpenGL.error.GLError: GLError(err = 12289,baseOperation = eglMakeCurrent ....

The bug of pyrender is really a pain in the ass...
``bash
# for centos 
sudo yum install libXext libSM libXrender freeglut-devel  
# for ubuntu: use pyrender in OSMesa mode  
sudo apt update  
sudo apt-get install libsm6 libxrender1 libfontconfig1 freeglut3-dev  
sudo apt --fix-broken install  
sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb  
sudo dpkg -i ./mesa_18.3.3-0.deb || true  
sudo apt install -f  
git clone https://github.com/mmatl/pyopengl.git  
pip install ./pyopengl  

## from OpenGL.osmesa import ( ImportError: cannot import name 'OSMesaCreateContextAttribs'

git clone git@github.com:mmatl/pyopengl.git

pip install ./pyopengl
