import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyicoshift",
    version="0.0.1",
    install_requires=[
        "numpy>=0.18.4",
        "nmrglue>=0.7",
        "scipy>=1.4.1"
    ],
    author="Sebastian Krossa",
    author_email="sebastian.krossa@ntnu.no",
    description="Python 3 Version of icoshift",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sekro/pyicoshift",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
