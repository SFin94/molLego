from setuptools import setup, find_packages
setup(
    name='molLego',
    author='Sophie Finnigan',
    url='https://github.com/SFin94/molLego',
    version="1.0",
    packages=find_packages(),
    install_requires=[
         'scipy',
         'matplotlib',
         'pandas',
         'seaborn',
         'numpy',
         'setuptools',
    ],
    python_requires='>=3.6'
)

