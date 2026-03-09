# 这玩意是我自己写的,原来没有

import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(
    name='isaaclab',
    version='0.0.0',
    packages=find_packages(),
    url='https://github.com/isaac-sim/IsaacLab',
    license='',
    author="Nvidia Omniverse",
    install_requires=[
        
    ],
)
