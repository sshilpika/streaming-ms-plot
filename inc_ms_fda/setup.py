import sys
import os
from distutils.core import setup

setup(name='inc-ms-fda',
      version='0.01',
      packages=[''],
      package_dir={'': '.'},
      install_requires=['numpy', 'scipy', 'scikit-fda', 'matplotlib'],
      py_modules=['inc_ms_fda'])
