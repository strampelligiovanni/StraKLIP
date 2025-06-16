# from setuptools import setup, find_packages
#
# setup(
#     name='StraKLIP',
#     version='1.1.1',
#     packages=find_packages(),
#     install_requires=[
#         # put your dependencies here, e.g. 'numpy', 'torch', ...
#     ],
# )

from setuptools import setup, find_packages

setup(packages=find_packages(),
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      )