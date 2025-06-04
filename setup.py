from setuptools import setup, find_packages

NAME = "nprba"
VERSION = "0.0.1"
AUTHORS = "Shamil Mamedov, Rene Geist"
MAINTAINER_EMAIL = ""
DESCRIPTION = "package for learning deformable linear objects dynamics using physics-informed neural networks"

setup(
    name=NAME,
    version=VERSION,
    author=AUTHORS,
    author_email=MAINTAINER_EMAIL,
    packages=find_packages()
)