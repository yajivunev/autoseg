from setuptools import setup

setup(
        name='autoseg',
        version='0.1',
        description='Modules and scripts for machine learning on EM images.',
        url='https://github.com/yajivunev/autoseg',
        author='Vijay Venu',
        author_email='vvenu@utexas.edu',
        packages=[
            'autoseg',
            'autoseg.segment'
        ],
        install_requires=[
            "torch",
            "numpy",
            "zarr",
            "scipy",
            "scikit-image",
            "gunpowder",
        ]
)
