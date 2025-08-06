from setuptools import setup, find_packages

setup(
    name='SBD',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    author='Matteo Zucchetta',
    author_email='matteo.zucchetta@cnr.it',
    description='Functionalities for tuning object detection models for detecting small boats in satellite images',
    license='BSD 2',
)