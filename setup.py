from setuptools import setup, find_packages

setup(
    name="mri-analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'Pillow>=8.0.0',
        'torch>=1.9.0',
    ],
    entry_points={
        'console_scripts': [
            'mri-analyzer=src.cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A CLI tool for analyzing MRI scan series using neural networks",
    keywords="mri, neural-network, medical-imaging",
    python_requires=">=3.7",
) 