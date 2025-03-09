from setuptools import setup, find_packages

setup(
    name='welqrate', 
    version='0.1.0',  
    description='The official implementation of the WelQrate dataset and benchmark',  
    url='https://github.com/xwang38438/WelQrate.git',  
    author='Xin Wang',  
    author_email='xin.wang.1@vanderbilt.edu',  
    packages=find_packages(include=["welqrate", "welqrate.*"]),  # Only include 'welqrate' package
    install_requires=[
        "matplotlib==3.7.1",
        "fsspec==2024.5.0",
        "networkx==2.8.7",
        "numpy==1.23",
        "pandas==1.4",
        "scipy==1.11.0", 
        "setuptools==68.0.0",
        "tqdm==4.65.0",
        "rdkit==2023.09.6",
        "optuna==4.2.1",
    ],  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  
)



# setup(
#     name='welqrate', 
#     version='0.1.0',  
#     description='The official implementation of the WelQrate dataset and benchmark',  
#     url='https://github.com/xwang38438/WelQrate.git',  
#     author='Xin Wang',  
#     author_email='xin.wang.1@vanderbilt.edu',  
#     packages=find_packages(),  
#     install_requires=[],  # List dependencies here if any #rdkit
#     classifiers=[
#         'Programming Language :: Python :: 3',
#         'License :: OSI Approved :: MIT License',  
#     ],
#     python_requires='>=3.8',  
# )
