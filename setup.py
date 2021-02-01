from setuptools import setup, find_packages

setup(
    name='copia',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/mikekestemont/copia',
    license='CC-BY',
    author='Mike Kestemont & Folgert Karsdorp',
    description='Bias correction for richness in abundance data',
    requires=['numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas', 'tqdm', 'pytest']
)
