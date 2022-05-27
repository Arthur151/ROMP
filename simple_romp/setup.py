import setuptools
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requireds = ["opencv-python","torch",
        'setuptools>=18.0.0',
        'cython',
        'numpy>=1.21.0',
        'typing-extensions>=4.1'
        'scipy',
        'lap']

setuptools.setup(
    name='simple_romp',
    version='1.0.5',
    author="Yu Sun",
    author_email="yusun@stu.hit.edu.cn",
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0.0',
        'cython',
        'numpy>=1.21.0',
        'typing-extensions>=4.1'
        'scipy',
        'lap'
    ],
    install_requires=requireds,
    description="ROMP: Monocular, One-stage, Regression of Multiple 3D People, ICCV21",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arthur151/ROMP",
    packages=[
        'romp',
        'vis_human',
        'vis_human.sim3drender',
        'vis_human.sim3drender.lib',
        'bev',
        'tracker',
    ],
    ext_modules=cythonize([Extension("Sim3DR_Cython",
                           sources=["vis_human/sim3drender/lib/rasterize.pyx",
                                    "vis_human/sim3drender/lib/rasterize_kernel.cpp"],
                           language='c++',
                           include_dirs=[numpy.get_include()],
                           extra_compile_args=["-std=c++11"])]),
    include_package_data=True,
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: Other/Proprietary License",
         "Operating System :: OS Independent",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/Arthur151/ROMP/issues",
    },
    entry_points={
        "console_scripts": [
            "romp=romp.main:main",
            "bev=bev.main:main",
        ],
    },
)
