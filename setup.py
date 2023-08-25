#!/usr/bin/env python3

import os
import sys

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    with open(os.path.join("tianshou", "__init__.py")) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
        return None


def get_readme() -> str:
    with open("README.md") as f:
        return f.read()


def get_install_requires() -> str:
    return [
        "gymnasium>=0.29.0",
        "tqdm",
        "numpy>1.16.0",  # https://github.com/numpy/numpy/issues/12793
        "tensorboard>=2.5.0",
        "torch>=2.0.1",
        "numba>=0.51.0",
        "h5py>=3.9.0",  # to match tensorflow's minimal requirements
        "packaging",
        "pettingzoo>=1.22",
    ]


def get_extras_require() -> str:
    req = {
        "dev": [
            "sphinx<7",
            "black>=23.7.0",
            "ruff>=0.0.285",
            "sphinx_rtd_theme",
            "jinja2",
            "sphinxcontrib-bibtex",
            "pytest",
            "pytest-cov",
            "ray>=1.0.0",
            "wandb>=0.12.0",
            "networkx",
            "mypy",
            "doc8",
            "scipy",
            "pillow",
            "pygame>=2.1.0",  # pettingzoo test cases pistonball
            "pymunk>=6.2.1",  # pettingzoo test cases pistonball
            "nni>=2.3,<3.0",  # expect breaking changes at next major version
            "gym>=0.22.0",
            "shimmy",
        ],
        "atari": ["atari_py", "opencv-python"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    }
    if sys.platform == "linux":
        req["dev"].append("envpool>=0.7.0")
    return req


setup(
    name="tianshou",
    version=get_version(),
    description="A Library for Deep Reinforcement Learning",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thu-ml/tianshou",
    author="TSAIL",
    author_email="trinkle23897@gmail.com",
    license="MIT",
    python_requires=">=3.9",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="reinforcement learning platform pytorch",
    packages=find_packages(exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]),
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
