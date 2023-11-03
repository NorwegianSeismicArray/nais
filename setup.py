
import setuptools
from setuptools import find_packages

with open("./README.md") as file:
    read_me_description = file.read()

setuptools.setup(
    name="nais",
    version="1.4",
    author="Erik B. Myklebust & Tord S. Stangeland",
    author_email="tord@norsar.no",
    description="NORSAR AI System.",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NorwegianSeismicArray/nais",
    download_url='https://github.com/NorwegianSeismicArray/nais/archive/refs/tags/1.0.tar.gz',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'tensorflow',
                      'keras_tuner',
                      'scikit-learn',
                      'pandas',
                      'tqdm',
                      'pillow'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
