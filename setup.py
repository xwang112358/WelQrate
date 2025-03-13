from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='welqrate', 
    version='0.1.3',  
    description='The official implementation of the WelQrate dataset and benchmark',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/xwang38438/WelQrate.git',  
    author='Xin Wang, Yunchao Liu',  
    author_email='xin.wang.1@vanderbilt.edu, yunchao.liu@vanderbilt.edu',  
    packages=find_packages(include=["welqrate", "welqrate.*"]),
    include_package_data=True,  # Add this if you have non-Python files
    install_requires=[
        "torch==2.1.0",
        "torch_geometric==2.3.1",
        "matplotlib==3.7.1",
        "networkx==2.8.7",
        "numpy==1.23",
        "pandas==1.4",
        "scipy==1.11.0", 
        "setuptools==68.0.0",
        "tqdm==4.65.0",
        "rdkit==2023.09.6",
        "optuna==4.2.1",
        "scikit-learn==1.6.1"
    ],
    dependency_links=[
        "https://data.pyg.org/whl/torch-2.1.0+cu121.html"
    ],
    extras_require={
        "pyg": ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  
)