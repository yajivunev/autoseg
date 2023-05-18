from setuptools import setup, find_packages

setup(
        name='autoseg',
        version='0.1',
        description='Modules and scripts for machine learning on EM images.',
        url='https://github.com/yajivunev/autoseg',
        author='Vijay Venu',
        author_email='vvenu@utexas.edu',
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            "torch",
            "numpy",
            "zarr",
            "scipy",
            "scikit-image",
            "gunpowder",
        ]
)
