import os

from setuptools import find_packages
from setuptools import setup


dir_path = os.path.dirname(os.path.realpath(__file__))

packages = find_packages('src')
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'realsense_lcm' or p.startswith('realsense_lcm.')

def pkg_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('../..', path, filename))
    return paths

setup(
    name='realsense_lcm',
    author='Anthony Simeonov',
    license='MIT',
    packages=packages,
    package_dir={'': 'src'},
)

