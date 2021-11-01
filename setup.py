
import setuptools

with open("README.md") as file:
    read_me_description = file.read()

setuptools.setup(
    name="nais",
    version="0.1.1",
    author="Erik B. Myklebust",
    author_email="erik@norsar.no",
    description="NORSAR AI System.",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    url="http://bitbucket:7990/projects/GEOB/repos/nais",
    packages=['nais'],
    install_requires=['tensorflow','numpy','kapre','tensorflow_addons'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
