from setuptools import setup, find_packages

setup(
    name='gfpgan_plus',
    version='0.1.0',
    description='A Python package for image generation using GANs.',
    author='Dat Viet Thanh Nguyen',
    author_email='thanhdatnv2712@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'opencv-python'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
