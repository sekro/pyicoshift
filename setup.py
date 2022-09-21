import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyicoshift",
    version="0.0.2",
    install_requires=[
        "numpy>=1.23.3",
        "nmrglue>=0.9",
        "matplotlib>=3.6.0",
        "scipy>=1.9.1",
        "scikit-learn>=1.1.2",
        "statsmodels>=0.13.2"
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
    python_requires='>=3.8',
)
