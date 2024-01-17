import io
import os

from setuptools import find_packages, setup


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="aillustrate",
    version="0.0.1",
    description="automatic image generation pipeline",
    url="https://github.com/Aillustrate/wonderslide-interior-generation/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Aillustrate team",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={"console_scripts": ["aillustrate = aillustrate.__main__:main"]},
)
