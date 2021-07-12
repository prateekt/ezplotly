import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EZPlotly",
    version="0.0.8",
    author="Prateek Tandon",
    author_email="prateek1.tandon@gmail.com",
    description="An easy wrapper for making Plotly plots in Jupyter notebooks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prateekt/EasyPlotly",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)