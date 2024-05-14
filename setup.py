import os

from setuptools import setup

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def load_requirements(root: str):
    path = os.path.join(root, "requirements.txt")
    with open(path) as f:
        return f.read().splitlines()


setup(
    name="chemicalspace",
    version="0.1",
    author="Giulio Mattedi",
    packages=["chemicalspace"],
    install_requires=load_requirements(LOCAL_PATH),
)
