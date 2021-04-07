from setuptools import setup, find_packages

setup(
    name='copia',
    version='0.1.1',
    packages=find_packages(),
    url='https://github.com/mikekestemont/copia',
    license='CC-BY',
    author='Mike Kestemont & Folgert Karsdorp',
    description='Bias correction for richness in abundance data',
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
