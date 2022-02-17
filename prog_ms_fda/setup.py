import sys
import os
from distutils.core import setup

setup(name='prog-ms-fda',
      version='0.02',
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'scikit-fda', 'matplotlib'],
      py_modules=['prog_ms_fda'])
