import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyicoshift-your-username",
    version="0.0.1",
    author="Sebastian Krossa",
    author_email="sebastian.krossa@ntnu.no",
    description="Python Version of icoshift",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sekro/pyicoshift",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)