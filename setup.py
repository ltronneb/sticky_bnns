from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='sazz',
    version='0.1',
    packages= find_packages("."),
    install_requires=requirements,
    url='',
    license='',
    author='leiv',
    author_email='ltronneb@math.uio.no',
    description=''
)
