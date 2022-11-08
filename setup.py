"""
qcten
Write me
"""
import sys
from setuptools import setup, find_packages
import versioneer

short_description = "Write me".split("\n")[0]

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = None

setup(
    name='qcten',
    author='Gosia Olejniczak',
    author_email='gosia.olejniczak@gmail.com',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',
    packages=find_packages(where="qcten"),
    setup_requires=[] + pytest_runner,
)
