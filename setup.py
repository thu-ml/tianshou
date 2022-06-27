#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("tianshou", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return [
        "gym>=0.23.1",
        "tqdm",
        "numpy>1.16.0",  # https://github.com/numpy/numpy/issues/12793
        "tensorboard>=2.5.0",
        "torch>=1.4.0",
        "numba>=0.51.0",
        "h5py>=2.10.0",  # to match tensorflow's minimal requirements
        "protobuf~=3.19.0",  # breaking change, sphinx fail
    ]


def get_extras_require() -> str:
    req = {
        "dev": [
            "sphinx<4",
            "sphinx_rtd_theme",
            "jinja2<3.1",  # temporary fix
            "sphinxcontrib-bibtex",
            "flake8",
            "flake8-bugbear",
            "yapf",
            "isort",
            "pytest",
            "pytest-cov",
            "ray>=1.0.0",
            "wandb>=0.12.0",
            "networkx",
            "mypy",
            "pydocstyle",
            "doc8",
            "scipy",
            "pillow",
            "pettingzoo>=1.17",
            "pygame>=2.1.0",  # pettingzoo test cases pistonball
            "pymunk>=6.2.1",  # pettingzoo test cases pistonball
            "nni>=2.3",
        ],
        "atari": ["atari_py", "opencv-python"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    }
    if sys.platform == "linux":
        req["dev"].append("envpool>=0.5.3")
    return req


setup(
    name="tianshou",
    version=get_version(),
    description="A Library for Deep Reinforcement Learning",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thu-ml/tianshou",
    author="TSAIL",
    author_email="trinkle23897@gmail.com",
    license="MIT",
    python_requires=">=3.6",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="reinforcement learning platform pytorch",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
