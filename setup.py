from setuptools import setup, find_packages
import os

VERSION = '0.0.1'
DESCRIPTION = 'Tensor-based autdiff engine and neural network API'
LONG_DESCRIPTION = '''
                        A package that allows one to perform tensor operations and 
                        obtain the derivatives of these tensors via reverse-mode 
                        automatic differentiation. A small neural network API is 
                        included via which one can build and train neural networks.
                    '''

# Setting up
setup(
    name="madnet",
    version=VERSION,
    author="Jordan Madden",
    author_email="<jordanmadden285@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'tensors', 'neural networks', 'automatic differentiation'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)