
import setuptools
from setuptools import find_packages

with open("./README.md") as file:
    read_me_description = file.read()

req = []
with open('./requirements.txt') as file:
    for l in file.readlines():
        req.append(l.strip())

setuptools.setup(
    name="nais",
    version="0.2.6",
    author="Erik B. Myklebust",
    author_email="erik@norsar.no",
    description="NORSAR AI System.",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NorwegianSeismicArray/nais",
    packages=find_packages(),
    install_requires=req,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
