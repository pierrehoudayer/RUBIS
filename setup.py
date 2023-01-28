from setuptools import setup, find_packages

setup(
    name='RUBIS',
    version='0.1.0',
    url='https://github.com/pierrehoudayer/RUBIS.git',
    author='Pierre Houdayer',
    author_email='pierre.houdayer@obspm.fr',
    packages=find_packages(),    
    install_requires=['numpy >= 1.20.3', 'matplotlib >= 3.4.2', 'scipy >= 1.7.3'],
)