from setuptools import setup, find_packages

setup(
    name='mp2bvh',
    version='0.2.4',
    packages=find_packages(include=['mp2bvh', 'mp2bvh.*']),

    entry_points={
        'console_scripts': ['mp2bvh=mp2bvh.main:main']
    }
)


