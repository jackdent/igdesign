from setuptools import find_packages, setup

setup(
    name="igdesign",
    version="1.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
