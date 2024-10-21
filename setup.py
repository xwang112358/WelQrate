from setuptools import setup, find_packages

setup(
    name='welqrate',  # Replace with your desired package name
    version='0.1.0',  # Version number of your package
    description='The official implementation of the WelQrate dataset and benchmark',  
    url='https://github.com/xwang38438/WelQrate.git',  
    author='Xin Wang',  # Your name
    author_email='xin.wang.1@vanderbilt.edu',  # Your email
    packages=find_packages(),  # Automatically finds all packages in your repo
    install_requires=[],  # List dependencies here if any
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Adjust license if needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Python version requirement
)
