from setuptools import setup, find_packages
import os

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

# Setting up
setup(
    name="mdgrad",
    version='0.3',
    author="Jordan Madden",
    author_email="<jordanmadden285@gmail.com>",
    description='Tensor-based autdiff engine and neural network API',
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'tensors', 'neural networks', 'automatic differentiation'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=3.8'
)