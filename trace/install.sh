# compile deformable convolution
cd lib/models/deform_conv
python setup.py develop
cd ../../..

# compile bounding box iou
cd lib/tracker/cython_bbox
python setup.py install
cd ../../..

#pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox