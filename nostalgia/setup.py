import os
from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def parse_requirements(requirements_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    requirements_file = os.path.join(project_root, requirements_path)
    with open(requirements_file) as f:
        return [line.strip() for line in f if line.strip()]

REQUIREMENTS = parse_requirements("nostalgia/requirements.txt")
ext_modules = [
    Pybind11Extension(
        'fingerprint_pybind',
        ['fingerprint_pybind.cpp', 'hash/sha1.c'],
        include_dirs=['hash'],
        libraries=['gomp'],
        extra_compile_args=['-std=c++17', '-O3', '-fopenmp', '-g'],
        extra_link_args=['-fopenmp'],
    )
]
setup(
    name='nostalgia',
    version='0.1.0',
    description='Audio fingerprinting system',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=REQUIREMENTS,
    zip_safe=False,
    python_requires=">=3.7",
)
