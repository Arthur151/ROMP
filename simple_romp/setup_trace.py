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
        'typing-extensions>=4.1',
        'scipy',
        'lap']

setuptools.setup(
    name='simple_romp',
    version='2.0.2',
    author="Yu Sun",
    author_email="yusunhit@gmail.com",
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0.0',
        'cython',
        'numpy>=1.21.0',
        'typing-extensions>=4.1',
        'scipy',
        'lap'],
    install_requires=requireds,
    description="ROMP [ICCV21], BEV [CVPR22], TRACE [CVPR23]",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Arthur151/ROMP",
    packages=[
        'romp',
        'vis_human',
        'vis_human.sim3drender',
        'vis_human.sim3drender.lib',
        'bev',
        'trace2',
        'trace2.tracker',
        'trace2.models',
        'trace2.models.raft',
        'trace2.models.raft.utils',
        'trace2.models.deform_conv',
        'trace2.models.deform_conv.functions',
        'trace2.models.deform_conv.modules',
        'trace2.results_parser',
        'trace2.evaluation',
        'trace2.evaluation.dynacam_evaluation',
        'trace2.evaluation.TrackEval',
        'trace2.evaluation.TrackEval.trackeval',
        'trace2.evaluation.TrackEval.trackeval.metrics',
        'trace2.evaluation.TrackEval.trackeval.datasets',
        'trace2.evaluation.TrackEval.trackeval.baselines',
        'trace2.utils',
        'tracker'],
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
            "trace2=trace2.main:main",
            "romp.prepare_smpl=romp.pack_smpl_info:main",
            "bev.prepare_smil=bev.pack_smil_info:main",
        ],
    },
)

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, sources):
    return CUDAExtension(
        name=name,
        sources=[p for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })

setuptools.setup(
        name='deform_conv',
        ext_modules=[
            make_cuda_ext(
                name='deform_conv_cuda',
                sources=[
                    'trace2/models/deform_conv/src/deform_conv_cuda.cpp',
                    'trace2/models/deform_conv/src/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_cuda',
                sources=[
                    'trace2/models/deform_conv/src/deform_pool_cuda.cpp',
                    'trace2/models/deform_conv/src/deform_pool_cuda_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)