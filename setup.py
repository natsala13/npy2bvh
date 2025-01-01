from setuptools import setup, find_packages

setup(
    name='mp2bvh',
    version='0.2.4',
    author='Nathan Sala',
    author_email='natouda@gmail.com',
    description='Convert between different representation of human motions.',
    packages=find_packages(include=['mp2bvh', 'mp2bvh.*']),
    classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': ['mp2bvh=mp2bvh.main:main']
    }
)
