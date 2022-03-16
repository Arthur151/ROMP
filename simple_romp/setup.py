import setuptools
from distutils.core import setup, Extension
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    import numpy
except ImportError:
    # create closure for deferred import
    def cythonize (*args, ** kwargs ):
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext
        return cythonize(*args, ** kwargs)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requireds = ["numpy","opencv-python","cython","torch"]

setuptools.setup(
    name='simple_romp',
    version='0.0.1',
    author="Yu Sun",
    author_email="yusun@stu.hit.edu.cn",
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
        'numpy',
    ],
    install_requires=requireds,
    description="ROMP: Monocular, One-stage, Regression of Multiple 3D People, ICCV21",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arthur151/ROMP",
    packages=[
        'romp',
        'sim3drender',
        'sim3drender.lib',
    ],
    # cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([Extension("sim3drender.Sim3DR_Cython",
                           sources=["sim3drender/lib/rasterize.pyx",
                                    "sim3drender/lib/rasterize_kernel.cpp"],
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
        ],
    },
)
