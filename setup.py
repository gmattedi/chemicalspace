import os

from setuptools import setup

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))


def load_requirements(file: str, root: str = LOCAL_PATH):
    path = os.path.join(root, file)
    with open(path) as f:
        return f.read().splitlines()


setup(
    name="chemicalspace",
    version="0.1",
    author="Giulio Mattedi",
    packages=["chemicalspace"],
    install_requires=load_requirements("requirements.txt"),
    extras_require={"dev": load_requirements("requirements-dev.txt")},
)
