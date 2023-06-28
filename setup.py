from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().split()

setup(
    name='grounding',
    version='1.1.0',
    description='Language grounding experiments on the Alfred dataset.',
    author='Amric Trudel',
    url="https://github.com/atrudel/alfred_grounding",
    python_requires='~=3.9',
    install_requires=requirements,
    packages=find_packages()
)