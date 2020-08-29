#！/bin/bash
pip install -r requirements.txt

pip uninstall smplx
cd ../models/smplx
python setup.py install 

cd ..
cd ../src
