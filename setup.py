import os
from setuptools import setup, find_packages


directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='copia',
    version='0.1.6',
    packages=find_packages(),
    url='https://github.com/mikekestemont/copia',
    license='CC-BY',
    author='Mike Kestemont & Folgert Karsdorp',
    description='Bias correction for richness in abundance data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'pytest>=7.4', 
        'scipy>=1.11',
        'numpy>=1.24',
        'pandas>=2.1',
        'matplotlib>=3.8',
        'tqdm>=4.64'
    ]
)
