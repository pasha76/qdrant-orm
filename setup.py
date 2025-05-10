"""
Setup script for Qdrant ORM
"""
from setuptools import setup, find_packages

setup(
    name="qdrant-orm",
    version="0.2.0",
    description="A lightweight SQLAlchemy-style ORM for Qdrant vector database",
    author="Tolga Gunduz",
    packages=find_packages(),
    install_requires=[
        "qdrant-client>=1.13.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
