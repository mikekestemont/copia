import os
from setuptools import setup, find_packages


directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='copia',
    version='0.1.3',
    packages=find_packages(),
    url='https://github.com/mikekestemont/copia',
    license='CC-BY',
    author='Mike Kestemont & Folgert Karsdorp',
    description='Bias correction for richness in abundance data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.1',
        'matplotlib>=3.3.2',
        'seaborn>=0.11.0',
        'pandas>=1.1.2',
        'tqdm>=4.48.0',
        'pytest>=6.2.2'
    ]
)
