from setuptools import setup, find_packages

setup(
    name="DisCoWebS",
    version="1.0.0",
    # Exclude test, build, and utils directories from being installed
    packages=find_packages(exclude=["build*", "test*", "utils*"]),
)
