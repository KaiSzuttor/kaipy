# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# Get current version
execfile('./kaipy/version.py')

setup(
    name='kaipy',
    version=__version__,
    description='Python package for data evaluation of simulations.', 
    url='https://github.com/KaiSzuttor/kaipy',
    author='Kai Szuttor',
    author_email='kaiszuttor@gmail.com',
    license='GNU General Public License',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    packages=find_packages('kaipy', exclude=['doc', 'tests*']),
    install_requires=['numpy>=1.9.2']
)
