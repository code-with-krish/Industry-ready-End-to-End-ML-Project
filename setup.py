from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:

    requirements = []

    with open(file_path) as file:
        for line in file:
            line = line.strip()

            if line and line != "-e .":
                requirements.append(line)

    return requirements


setup(
    name="End-TO-ENd-ML-Project",
    version="0.0.1",
    author="Krishna Kanta Maiti",
    author_email="krishnakantamaiti7337@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)